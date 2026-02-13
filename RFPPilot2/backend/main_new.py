from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import threading
import uuid
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import re
import hashlib
from datetime import datetime
import base64
import os
from dotenv import load_dotenv
import smtplib
from email.message import EmailMessage
from email.utils import formataddr
import mimetypes

from pydantic import BaseModel

# Load environment variables
load_dotenv()

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
    clear_all,
    add_gmail_email,
    list_gmail_emails,
    add_notification,
    list_notifications,
    get_unprocessed_gmail_emails,
    mark_gmail_email_processed,
    clear_notifications,
    clear_gmail_data
)

# Pydantic models for request bodies
class ProposalEditRequest(BaseModel):
    updated_bid: str

class CoverLetterRequest(BaseModel):
    buyer: str
    rfp_title: str
    rfp_id: str = ""

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
    
    print("[Warmup] Starting browser warmup (persistent page init)...")
    try:
        # Initialize persistent page in THIS thread (worker thread) where it will be used
        # This avoids greenlet thread conflicts and warms up the browser
        page = rfp_module._init_browser()  # Returns persistent page
        page.goto("about:blank")
        print("[Warmup] ✅ Persistent browser page initialized and ready")
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
    # 🚀 EAGER MODULE LOAD: Ensure embeddings loaded before declaring startup complete
    print("[Startup] 🚀 Loading RFP module with global initializations...")
    
    # Load module synchronously to ensure embeddings initialization complete
    # Browser page will lazy-load on first use in worker thread (persistent page pattern)
    rfp_module = load_rfp_module()
    
    # Verify critical components are ready
    if rfp_module.SKU_EMBEDDINGS is not None:
        print(f"[Startup] ✅ SKU embeddings ready ({len(rfp_module.SKU_EMBEDDINGS)} embeddings)")
    else:
        print("[Startup] ⚠️ SKU embeddings not loaded")
    
    print("[Startup] ✅ Persistent browser page will initialize on first request (lazy-load)")
    
    # Fire-and-forget warmup for additional checks
    _pipeline_executor.submit(warmup_pipeline)
    print("[Startup] 🚀 Application startup complete - embeddings pre-loaded, persistent page pattern enabled")


# Background runner
def run_pipeline_bg(task_id: str):
    try:
        task_status[task_id] = "running"
        clear_all()
        
        # Use pre-loaded module (already initialized)
        rfp_module = load_rfp_module()
        
        state = {}
        
        # Sales Agent (process website PDFs first)
        state = rfp_module.sales_agent_node(state)
        sales_data = state.get("sales_output", [])

        # ------------------------------------------------------------------
        # Gmail Integration: Delegate to AI module for robust processing
        # - Removes manual regex extraction
        # - Uses LLM-based extraction with Ground Truth validation
        # - Ensures domain-intelligent matching for MEDC power cables
        # ------------------------------------------------------------------
        gmail_emails = get_unprocessed_gmail_emails()
        gmail_sales_data = []
        gmail_extracted_items_map = {}  # Map rfp_id -> extracted_item_list
        gmail_text_cache = {}  # Map rfp_id -> full text for Technical Agent

        if gmail_emails:
            print(f"[Gmail Integration] Processing {len(gmail_emails)} Gmail emails through AI module")
            for email_obj in gmail_emails:
                message_id = str(email_obj.get("message_id") or "").strip()
                subject = str(email_obj.get("subject") or "").strip()
                body_text = str(email_obj.get("body_text") or "").strip()
                
                if not subject and not body_text:
                    continue
                
                # Combine email content for AI processing
                combined_text = f"Subject: {subject}\n\n{body_text}"
                
                # Create state for this Gmail RFP (AI module handles extraction, matching, domain detection)
                gmail_state = {
                    "gmail_rfp_text": combined_text,
                    "gmail_message_id": message_id
                }
                
                # Process through AI module (uses LLM + Ground Truth logic)
                try:
                    gmail_state = rfp_module.sales_agent_node(gmail_state)
                    gmail_rfps = gmail_state.get("sales_output", [])
                    
                    # Debug: Check what we got
                    print(f"[Gmail Integration] 🔍 Debug: gmail_rfps count={len(gmail_rfps)}")
                    
                    # Add Gmail-specific metadata
                    for rfp in gmail_rfps:
                        rfp["source"] = "gmail"
                        rfp["gmail_message_id"] = message_id
                        rfp["gmail_received_timestamp"] = email_obj.get("received_timestamp", "")
                        
                        # Extract items from the RFP object (not from state root)
                        rfp_id = rfp.get("rfp_id")
                        extracted_items = rfp.get("extracted_item_list", [])
                        
                        print(f"[Gmail Integration] 🔍 Debug: RFP ID={rfp_id}, extracted_items count={len(extracted_items)}")
                        
                        # Store extracted items and text cache for this RFP
                        if rfp_id:
                            if extracted_items:
                                gmail_extracted_items_map[rfp_id] = extracted_items
                                gmail_text_cache[rfp_id] = combined_text
                                print(f"[Gmail Integration] 📦 Stored {len(extracted_items)} items for {rfp_id}")
                            else:
                                print(f"[Gmail Integration] ⚠️ No extracted_items for {rfp_id} (empty list)")
                        else:
                            print(f"[Gmail Integration] ⚠️ RFP has no rfp_id!")
                    
                    gmail_sales_data.extend(gmail_rfps)
                    print(f"[Gmail Integration] ✅ Processed: {subject[:60]}")
                    
                except Exception as e:
                    print(f"[Gmail Integration] ⚠️ Error processing {message_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

        # Merge Gmail and website RFPs
        if gmail_sales_data:
            print(f"[Gmail Integration] Merged {len(gmail_sales_data)} Gmail RFPs with {len(sales_data)} website RFPs")
            sales_data = list(sales_data) + gmail_sales_data
            state["sales_output"] = sales_data
            
            # Store Gmail-specific data in state for later retrieval
            state["gmail_extracted_items_map"] = gmail_extracted_items_map
            state["gmail_text_cache"] = gmail_text_cache

        save_sales_output(sales_data)
        
        # Priority Selector (internal - doesn't affect saved sales output)
        state = rfp_module.select_highest_priority_rfp(state)
        
        # Restore Gmail-specific data for the selected RFP
        # After select_highest_priority_rfp, state["sales_output"] contains only the selected RFP
        selected_rfps = state.get("sales_output", [])
        selected_rfp_id = selected_rfps[0].get("rfp_id") if selected_rfps else None
        
        print(f"[Gmail Integration] 🔍 Debug: Selected RFP ID={selected_rfp_id}")
        print(f"[Gmail Integration] 🔍 Debug: gmail_extracted_items_map keys={list(state.get('gmail_extracted_items_map', {}).keys())}")
        
        if selected_rfp_id:
            gmail_extracted_items_map = state.get("gmail_extracted_items_map", {})
            gmail_text_cache = state.get("gmail_text_cache", {})
            
            # If selected RFP is a Gmail RFP, restore its extracted items
            if selected_rfp_id in gmail_extracted_items_map:
                state["extracted_item_list"] = gmail_extracted_items_map[selected_rfp_id]
                state["gmail_rfp_text"] = gmail_text_cache.get(selected_rfp_id, "")
                print(f"[Gmail Integration] 🔄 Restored {len(state['extracted_item_list'])} items for selected RFP {selected_rfp_id}")
            else:
                print(f"[Gmail Integration] ⚠️ Selected RFP {selected_rfp_id} not found in gmail_extracted_items_map")
        
        # Master Agent
        state = rfp_module.master_agent_node(state)
        save_master_output(state.get("master_output", {}))
        
        # Debug: Check state before Technical Agent
        print(f"[Pre-Technical] 🔍 Debug: extracted_item_list count={len(state.get('extracted_item_list', []))}")
        print(f"[Pre-Technical] 🔍 Debug: gmail_rfp_text present={bool(state.get('gmail_rfp_text'))}")
        print(f"[Pre-Technical] 🔍 Debug: pdf_path={state.get('pdf_path')}")
        
        # Technical Agent
        state = rfp_module.technical_agent_node(state)
        # Get both formatted and raw outputs
        tech_formatted = state.get("technical_output", {})
        tech_raw = state.get("technical_output_raw", {})
        
        # Ensure technical_output_raw exists in state (required by proposal_builder)
        if not tech_raw and tech_formatted:
            # Fallback: use formatted as raw if raw doesn't exist
            state["technical_output_raw"] = tech_formatted
            tech_raw = tech_formatted
            print("[Technical Agent] ⚠️ Using formatted output as fallback for raw output")
        
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


class GmailRfpEvent(BaseModel):
    # New schema (preferred)
    message_id: Optional[str] = None
    sender: str
    subject: str
    body_text: Optional[str] = ""
    snippet: Optional[str] = ""
    received_timestamp: Optional[str] = None

    # Backward-compatible fields from older listener payloads
    timestamp: Optional[str] = None


@app.post("/integrations/gmail/rfp-event")
def ingest_gmail_rfp_event(event: GmailRfpEvent):
    from datetime import datetime, timezone
    # Store raw email (deduped) + create a notification. Do NOT trigger agents.
    if not event.received_timestamp and not event.timestamp:
        # If no timestamp provided, use current UTC time
        received_ts = datetime.now(timezone.utc).isoformat()
    else:
        received_ts = event.received_timestamp or event.timestamp
    
    body_text = event.body_text or ""
    snippet = event.snippet or ""
    
    print(f"[Gmail Ingest] Received email: '{event.subject}' from {event.sender}")
    print(f"[Gmail Ingest] Timestamp: {received_ts}")

    message_id = (event.message_id or "").strip()
    if not message_id:
        basis = f"{event.sender}|{event.subject}|{received_ts}|{body_text}"
        message_id = "legacy-" + hashlib.md5(basis.encode("utf-8", errors="ignore")).hexdigest()[:16]

    raw_email = {
        "message_id": message_id,
        "sender": event.sender,
        "subject": event.subject,
        "body_text": body_text,
        "snippet": snippet,
        "received_timestamp": received_ts,
    }

    inserted = add_gmail_email(raw_email)
    print(f"[Gmail Ingest] Stored in backend: {inserted} (False if duplicate)")
    if inserted:
        # Minimal notification schema for UI - store full body for View More
        notif = {
            "id": str(uuid.uuid4()),
            "type": "discovery",
            "title": event.subject if event.subject else "No Subject",
            "description": f"From: {event.sender}",
            "body": body_text,
            "snippet": snippet,
            "received_timestamp": received_ts,
            "read": False,
            "source": "gmail",
            "message_id": message_id,
        }
        add_notification(notif)
    return {"status": "ok", "inserted": inserted}


@app.get("/notifications")
def get_notifications():
    # Return newest first
    notifs = list_notifications()
    def _ts(n: Dict[str, Any]):
        return str(n.get("received_timestamp") or "")
    notifs.sort(key=_ts, reverse=True)
    return notifs

@app.delete("/notifications")
def clear_all_notifications():
    """Clear all notifications"""
    clear_notifications()
    return {"status": "cleared"}

@app.delete("/gmail-data")
def reset_gmail_data():
    """Clear all Gmail emails, processed IDs, and notifications (useful for testing)"""
    clear_gmail_data()
    return {"status": "cleared", "message": "All Gmail data and notifications cleared"}

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

@app.post("/proposal/edit")
def edit_proposal(request: ProposalEditRequest):
    """Update the proposal with edited content"""
    try:
        save_proposal_output(request.updated_bid)
        return {
            "status": "success",
            "message": "Proposal updated successfully",
            "updated_content": request.updated_bid
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to update proposal: {str(e)}"
        }

@app.post("/proposal/generate-cover-letter")
def generate_cover_letter(request: CoverLetterRequest):
    """Generate AI-powered cover letter using Groq API"""
    try:
        import requests
        
        GROQ_API_KEY = os.getenv("GROQ_API_KEY")
        GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
        
        if not GROQ_API_KEY:
            return {
                "status": "error",
                "message": "GROQ_API_KEY not configured in .env file"
            }
        
        # Get the actual generated bid proposal text
        bid_proposal = get_proposal_output()
        
        if not bid_proposal or len(bid_proposal.strip()) < 50:
            return {
                "status": "error",
                "message": "No proposal found. Please generate the proposal first by running the Technical and Pricing agents."
            }
        
        # Construct a professional prompt using the actual proposal content
        prompt = f"""You are a professional bid writer. Below is a complete tender proposal that has been generated for a client.

PROPOSAL CONTENT:
{bid_proposal}

Based ONLY on the above proposal content, create a professional 3-paragraph cover letter that:
1. First paragraph: Express gratitude for the opportunity and reference the specific RFP title and buyer mentioned in the proposal
2. Second paragraph: Highlight expertise in the specific domain/materials mentioned in the proposal (e.g., if proposal is about paints, mention painting expertise; if about cables, mention cable expertise), emphasizing technical capability and compliance
3. Third paragraph: Convey confidence in meeting requirements, timeline commitment, and eagerness for partnership

IMPORTANT: Extract the buyer name, RFP title, and domain/materials from the proposal above. Use those exact details in the cover letter.

Tone: Professional, confident, client-focused
Length: 150-200 words total
Format: Plain text, no special formatting or markdown
Do NOT use placeholder company names - write as "We" or "Our organization"."""
        
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional bid writer with expertise in government and enterprise tender responses. Always read the proposal carefully and create a cover letter that matches its content."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 500
        }
        
        response = requests.post(GROQ_API_URL, json=payload, headers=headers, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            cover_letter = result["choices"][0]["message"]["content"].strip()
            
            # Try to extract buyer and title from proposal for response
            buyer = "Valued Client"
            rfp_title = "the tender"
            try:
                lines = bid_proposal.split('\n')
                for line in lines[:10]:  # Check first 10 lines
                    if "We thank" in line and "for inviting bids" in line:
                        parts = line.split("We thank ")[1].split(" for inviting")
                        buyer = parts[0].strip()
                    if "Subject:" in line or "Proposal for" in line:
                        if "RFP" in line:
                            title_part = line.split("Proposal for ")[1].split(" (RFP")[0] if "Proposal for" in line else ""
                            if title_part:
                                rfp_title = title_part.strip()
            except:
                pass
            
            return {
                "status": "success",
                "cover_letter": cover_letter,
                "buyer": buyer,
                "rfp_title": rfp_title
            }
        else:
            error_msg = response.text
            print(f"[Groq API Error] Status {response.status_code}: {error_msg}")
            return {
                "status": "error",
                "message": f"Groq API error: {error_msg}"
            }
            
    except Exception as e:
        print(f"[Cover Letter Generation Error] {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to generate cover letter: {str(e)}"
        }

@app.post("/proposal/send-email")
async def send_email(
    to_email: str = Form(...),
    subject: str = Form(...),
    html_content: str = Form(...),
    attachments: List[UploadFile] = File(default=[])
):
    """Send proposal email via Gmail SMTP with attachments"""
    try:
        # Gmail SMTP Configuration
        SENDER_EMAIL = os.getenv("GMAIL_SENDER_EMAIL", "sohamghorpade912@gmail.com")
        SENDER_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")  # Gmail App Password
        SMTP_HOST = "smtp.gmail.com"
        SMTP_PORT = 587
        
        if not SENDER_PASSWORD:
            return {
                "status": "error",
                "message": "GMAIL_APP_PASSWORD not configured in .env file"
            }
        
        print(f"[Gmail SMTP] Sender: {SENDER_EMAIL}")
        print(f"[Gmail SMTP] Recipient: {to_email}")
        print(f"[Gmail SMTP] Subject: {subject}")
        
        # Create email message
        msg = EmailMessage()
        msg['From'] = formataddr(("RFP Bidding System", SENDER_EMAIL))
        msg['To'] = to_email
        msg['Subject'] = subject
        
        # Set HTML content
        msg.set_content(html_content, subtype='html')
        
        # Process attachments
        for file in attachments:
            if file.filename:
                # Read file content
                file_content = await file.read()
                
                # Guess the MIME type
                mime_type, _ = mimetypes.guess_type(file.filename)
                if mime_type is None:
                    mime_type = 'application/octet-stream'
                
                # Split MIME type into maintype and subtype
                maintype, subtype = mime_type.split('/', 1)
                
                # Add attachment
                msg.add_attachment(
                    file_content,
                    maintype=maintype,
                    subtype=subtype,
                    filename=file.filename
                )
                print(f"[Gmail SMTP] Attached: {file.filename} ({mime_type})")
        
        # Connect to Gmail SMTP server and send
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()  # Secure the connection
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"[Gmail SMTP] ✅ Email sent successfully to {to_email}")
        
        return {
            "status": "success",
            "message": f"Email successfully sent via Gmail SMTP to {to_email}"
        }
        
    except smtplib.SMTPAuthenticationError as e:
        error_msg = str(e)
        print(f"[Gmail SMTP Error] Authentication failed: {error_msg}")
        return {
            "status": "error",
            "message": "Gmail authentication failed. Please verify the sender email and app password are correct."
        }
    except smtplib.SMTPException as e:
        error_msg = str(e)
        print(f"[Gmail SMTP Error] SMTP error: {error_msg}")
        return {
            "status": "error",
            "message": f"SMTP error: {error_msg}"
        }
    except Exception as e:
        error_msg = str(e)
        print(f"[Email Send Error] {error_msg}")
        return {
            "status": "error",
            "message": f"Failed to send email: {error_msg}"
        }
