from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import uuid
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from store_new import (
    save_sales_output,
    get_sales_output,
    save_master_output,
    get_master_output,
    save_technical_output,
    get_technical_output,
    save_pricing_output,
    get_pricing_output,
    save_compliance_output,
    get_compliance_output,
    save_proposal_output,
    get_proposal_output,
    clear_all
)

app = FastAPI(title="Agentic RFP Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(title="Agentic RFP Backend")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Task status store
task_status: Dict[str, str] = {}

# Dedicated single-thread executor for pipeline (avoids Playwright greenlet conflicts)
# All pipeline operations run in the same thread where Playwright browser is initialized
_pipeline_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pipeline")
_rfp_module = None

def load_rfp_module():
    """Load RFP module in pipeline thread (called once at startup)"""
    global _rfp_module
    if _rfp_module is None:
        import wonky_SBERT_copilot as rfp_module
        _rfp_module = rfp_module
        print("[Pipeline Thread] ✅ RFP module loaded")
    return _rfp_module

def warmup_pipeline():
    """Warm up browser, embeddings, and MongoDB to eliminate first-request latency"""
    rfp_module = load_rfp_module()
    
    print("[Warmup] Starting browser warmup...")
    try:
        # Warm up Playwright browser (actually open and close a page)
        page = rfp_module._BROWSER.new_page()
        page.goto("about:blank")
        page.close()
        print("[Warmup] ✅ Browser warmed up")
    except Exception as e:
        print(f"[Warmup] ⚠️ Browser warmup failed: {e}")
    
    print("[Warmup] Starting embedding model warmup...")
    try:
        # Warm up embedding model with test inference
        test_text = "Test warmup query for embedding model initialization"
        _ = rfp_module.embed_text(test_text)
        print("[Warmup] ✅ Embedding model warmed up")
    except Exception as e:
        print(f"[Warmup] ⚠️ Embedding warmup failed: {e}")
    
    print("[Warmup] Starting MongoDB warmup...")
    try:
        # Warm up MongoDB connection
        _ = list(rfp_module.sku_collection.find().limit(1))
        print("[Warmup] ✅ MongoDB connection warmed up")
    except Exception as e:
        print(f"[Warmup] ⚠️ MongoDB warmup failed: {e}")

@app.on_event("startup")
async def startup_event():
    """Pre-load RFP module and warm up all components in dedicated pipeline thread"""
    future = _pipeline_executor.submit(warmup_pipeline)
    future.result(timeout=45)  # Allow more time for warmup
    print("[Startup] ✅ Pipeline fully warmed up and ready")

# Background runner
def run_pipeline_bg(task_id: str):
    try:
        task_status[task_id] = "running"
        clear_all()
        
        # Use pre-loaded module (already initialized)
        rfp_module = load_rfp_module()
        
        state = {}
        
        # Sales Agent
        state = rfp_module.sales_agent_node(state)
        sales_data = state.get("sales_output", [])
        save_sales_output(sales_data)
        
        # Priority Selector (internal - doesn't affect saved sales output)
        state = rfp_module.select_highest_priority_rfp(state)
        
        # Master Agent
        state = rfp_module.master_agent_node(state)
        save_master_output(state.get("master_output", {}))
        
        # Technical Agent
        state = rfp_module.technical_agent_node(state)
        # Get both formatted and raw outputs
        tech_formatted = state.get("technical_output", {})
        tech_raw = state.get("technical_output_raw", {})
        
        # Build comparison matrices for each item
        if tech_raw.get("items") and tech_formatted.get("items"):
            for idx, formatted_item in enumerate(tech_formatted["items"]):
                if idx < len(tech_raw["items"]):
                    try:
                        comparison_matrix = rfp_module.build_comparison_matrix(
                            tech_raw["items"][idx],
                            domain=tech_raw.get("domain")
                        )
                        formatted_item["comparison_matrix"] = comparison_matrix
                    except Exception as e:
                        print(f"[Error building comparison matrix for item {idx}]: {e}")
        
        save_technical_output(tech_formatted)
        
        # Pricing Agent
        state = rfp_module.pricing_agent_node(state)
        save_pricing_output(state.get("pricing_output", {}))
        save_compliance_output(state.get("product_compliance_table", []))
        
        # Proposal Builder
        state = rfp_module.proposal_builder_node(state)
        save_proposal_output(state.get("final_bid_prose", ""))
        
        task_status[task_id] = "completed"
        
    except Exception as e:
        task_status[task_id] = "failed"
        print(f"[Pipeline Error] {e}")
        import traceback
        traceback.print_exc()

# POST /run
@app.post("/run")
def run_pipeline():
    task_id = str(uuid.uuid4())
    task_status[task_id] = "queued"
    
    # Submit to dedicated pipeline executor (same thread as Playwright initialization)
    _pipeline_executor.submit(run_pipeline_bg, task_id)
    
    return {"task_id": task_id, "status": "started"}

# GET /run/status/{task_id}
@app.get("/run/status/{task_id}")
def get_run_status(task_id: str):
    return {"task_id": task_id, "status": task_status.get(task_id, "unknown")}

# DELETE /rfps
@app.delete("/rfps")
def clear_rfps():
    clear_all()
    return {"status": "cleared"}

# SALES
@app.get("/rfps")
def list_rfps():
    return get_sales_output()

@app.get("/rfps/{rfp_id:path}")
def get_rfp_details(rfp_id: str):
    sales = get_sales_output()
    for rfp in sales:
        if str(rfp.get("rfp_id")) == rfp_id:
            return rfp
    return {"error": f"RFP {rfp_id} not found"}

# MASTER
@app.get("/master/output")
def get_master():
    return get_master_output()

# TECHNICAL
@app.get("/technical/output")
def get_technical():
    return get_technical_output()

# PRICING
@app.get("/pricing/output")
def get_pricing():
    return get_pricing_output()

@app.get("/pricing/totals")
def get_pricing_totals():
    pricing = get_pricing_output()
    return {
        "material_total": pricing.get("material_total", 0),
        "testing_total": pricing.get("testing_total", 0),
        "grand_total": pricing.get("grand_total", 0)
    }

# COMPLIANCE
@app.get("/compliance")
def get_compliance():
    return get_compliance_output()

# PROPOSAL
@app.get("/proposal")
def get_proposal():
    return {"proposal": get_proposal_output()}
