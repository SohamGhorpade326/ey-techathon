from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import threading
import uuid
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import re
import hashlib
from datetime import datetime

from pydantic import BaseModel

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
    list_notifications
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
    # Fire-and-forget warmup
    _pipeline_executor.submit(warmup_pipeline)
    print("[Startup] Pipeline warmup started in background")


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

        # ------------------------------------------------------------------
        # Gmail integration layer (NO agent modifications)
        # - Convert stored Gmail emails -> RFP objects in the same schema that
        #   downstream nodes expect (i.e., Sales Agent output schema)
        # - Merge with website-derived RFPs
        # ------------------------------------------------------------------
        sales_data = state.get("sales_output", [])

        def _stable_id_fallback(message_id: str, subject: str, body_text: str) -> str:
            basis = (message_id or "") + "|" + (subject or "") + "|" + (body_text or "")
            digest = hashlib.md5(basis.encode("utf-8", errors="ignore")).hexdigest()[:10]
            return f"GMAIL-{digest}"

        _MONTHS = {
            "jan": 1,
            "january": 1,
            "feb": 2,
            "february": 2,
            "mar": 3,
            "march": 3,
            "apr": 4,
            "april": 4,
            "may": 5,
            "jun": 6,
            "june": 6,
            "jul": 7,
            "july": 7,
            "aug": 8,
            "august": 8,
            "sep": 9,
            "sept": 9,
            "september": 9,
            "oct": 10,
            "october": 10,
            "nov": 11,
            "november": 11,
            "dec": 12,
            "december": 12,
        }

        def _normalize_deadline_for_priority(deadline_str: Optional[str]) -> Optional[str]:
            if not deadline_str:
                return None
            s = str(deadline_str).strip()
            if not s:
                return None

            # Normalize common punctuation/spaces.
            s = s.replace("\u2013", "-").replace("\u2014", "-")
            s = s.replace("|", " ")
            s = re.sub(r"\s+", " ", s).strip()

            # Prefer extracting a date/time substring if the line contains extra words.
            # Supports:
            # - 20 Feb 2026, 5:00 PM IST
            # - 20 Feb 2026
            # - 20/02/2026 17:00
            # - 20-02-2026
            month_names = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"

            # 1) Day Month Year (optionally time)
            m = re.search(
                rf"\b(\d{{1,2}})\s*({month_names})\s*(\d{{2,4}})"
                rf"(?:\s*,?\s*(\d{{1,2}})(?::(\d{{2}}))?(?:\s*([AP]M|A\.?M\.?|P\.?M\.?))?)?\b",
                s,
                flags=re.IGNORECASE,
            )
            if m:
                day = int(m.group(1))
                mon = _MONTHS.get(m.group(2).lower()[:3]) or _MONTHS.get(m.group(2).lower())
                year = int(m.group(3))
                if year < 100:
                    year = 2000 + year
                hour = m.group(4)
                minute = m.group(5)
                ampm_raw = m.group(6)
                if mon:
                    if hour:
                        hh = int(hour)
                        mm = int(minute) if minute else 0
                        # If AM/PM provided, emit 12-hour format expected by downstream parser.
                        if ampm_raw and re.search(r"p", ampm_raw, flags=re.IGNORECASE):
                            ampm = "PM"
                            return f"{day:02d}/{mon:02d}/{year} {hh:02d}:{mm:02d} {ampm}"
                        if ampm_raw and re.search(r"a", ampm_raw, flags=re.IGNORECASE):
                            ampm = "AM"
                            return f"{day:02d}/{mon:02d}/{year} {hh:02d}:{mm:02d} {ampm}"
                        # No AM/PM => emit 24-hour.
                        return f"{day:02d}/{mon:02d}/{year} {hh:02d}:{mm:02d}"
                    return f"{day:02d}/{mon:02d}/{year}"

            # 2) Numeric date (optionally time)
            m = re.search(
                r"\b(\d{1,2})[\-/\.](\d{1,2})[\-/\.](\d{2,4})"
                r"(?:\s*,?\s*(\d{1,2})(?::(\d{2}))?(?:\s*([AP]M))?)?\b",
                s,
                flags=re.IGNORECASE,
            )
            if m:
                day = int(m.group(1))
                mon = int(m.group(2))
                year = int(m.group(3))
                if year < 100:
                    year = 2000 + year
                hour = m.group(4)
                minute = m.group(5)
                ampm = (m.group(6) or "").upper()
                if hour:
                    hh = int(hour)
                    mm = int(minute) if minute else 0
                    if ampm in ["AM", "PM"]:
                        return f"{day:02d}/{mon:02d}/{year} {hh:02d}:{mm:02d} {ampm}"
                    return f"{day:02d}/{mon:02d}/{year} {hh:02d}:{mm:02d}"
                return f"{day:02d}/{mon:02d}/{year}"

            # Fallback: return as-is; downstream may still parse.
            return s

        def _extract_after_label(text: str, label: str) -> Optional[str]:
            # Match "Label: value" or "Label - value"; also handle value on next line.
            if not text:
                return None
            pat = rf"(?im)^\s*{re.escape(label)}\s*[:\-\u2013\u2014]\s*(.*)$"
            m = re.search(pat, text)
            if not m:
                return None
            v = (m.group(1) or "").strip()
            if v:
                return v
            # If label line has no inline value, take next non-empty line.
            start = m.end()
            tail = text[start:]
            for ln in tail.splitlines():
                ln = ln.strip()
                if ln:
                    # Stop if we hit another label-looking line.
                    if re.match(r"^[A-Za-z][A-Za-z /_-]{2,40}\s*[:\-\u2013\u2014]", ln):
                        break
                    return ln
            return None

        def _extract_first_matching_line_value(text: str, labels: List[str]) -> Optional[str]:
            for lab in labels:
                v = _extract_after_label(text, lab)
                if v:
                    return v
            return None

        def _extract_deadline_anywhere(text: str) -> Optional[str]:
            if not text:
                return None
            labels = [
                "Submission Deadline",
                "Bid Submission Deadline",
                "Deadline",
                "Due Date",
                "Last date of submission",
                "Last Date of Submission",
                "Last date of Submission",
            ]
            v = _extract_first_matching_line_value(text, labels)
            if v:
                return v

            # Search within same line as common phrases.
            month_names = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t)?(?:ember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
            # e.g. "Last date of submission is 20 Feb 2026, 5 PM IST"
            m = re.search(
                rf"(?im)(submission\s+deadline|bid\s+submission\s+deadline|deadline|due\s+date|last\s+date\s+of\s+submission)[^\n]{{0,80}}?"
                rf"(\d{{1,2}}\s*{month_names}\s*\d{{2,4}}(?:\s*,?\s*\d{{1,2}}(?::\d{{2}})?\s*(?:A\.?M\.?|P\.?M\.?|AM|PM)?)?)",
                text,
                flags=re.IGNORECASE,
            )
            if m:
                return m.group(2).strip()

            # Numeric date fallback near keywords.
            m = re.search(
                r"(?im)(submission\s+deadline|bid\s+submission\s+deadline|deadline|due\s+date|last\s+date\s+of\s+submission)[^\n]{0,80}?"
                r"(\d{1,2}[\-/\.]\d{1,2}[\-/\.]\d{2,4}(?:\s+\d{1,2}:\d{2}(?:\s*[AP]M)?)?)",
                text,
            )
            if m:
                return m.group(2).strip()

            return None

        def _extract_estimated_value_anywhere(text: str) -> Optional[str]:
            if not text:
                return None
            labels = [
                "Estimated Value",
                "Estimated Project Value",
                "Project Value",
                "Tender Value",
                "Estimated Cost",
            ]
            v = _extract_first_matching_line_value(text, labels)
            if v:
                return v

            # Currency range patterns.
            # Examples:
            # - INR 2.5 – 3.2 Crore
            # - ₹2.5 Cr
            # - Rs. 250 Lakh
            cur = r"(?:₹|INR|Rs\.?|Rupees)"
            num = r"\d{1,3}(?:[\d,]*\d)?(?:\.\d+)?"
            rng = rf"{num}(?:\s*(?:-|\u2013|to)\s*{num})?"
            unit = r"(?:crore|crores|cr\b|lakh|lakhs|lac|lacs)"
            m = re.search(rf"(?im)\b{cur}\s*{rng}\s*(?:{unit})\b", text)
            if m:
                return m.group(0).strip()

            # Sometimes unit precedes currency wording; try without currency.
            m = re.search(rf"(?im)\b{rng}\s*(?:{unit})\b", text)
            if m:
                return m.group(0).strip()

            return None

        def _value_exceeds_2cr(estimated_value: str) -> bool:
            if not estimated_value:
                return False
            s = str(estimated_value).lower()
            s = s.replace("\u2013", "-").replace("\u2014", "-")
            nums = [float(x.replace(",", "")) for x in re.findall(r"\d+(?:\.\d+)?", s)]
            if not nums:
                return False
            hi = max(nums)
            # If value is in crores/cr.
            if re.search(r"\b(crore|crores|cr)\b", s):
                return hi >= 2.0
            # If value is in lakhs (200 lakhs == 2 cr).
            if re.search(r"\b(lakh|lakhs|lac|lacs)\b", s):
                return hi >= 200.0
            return False

        def _compute_gmail_priority(deadline_str: Optional[str], estimated_value: Optional[str]) -> str:
            # Rule:
            # - Deadline in past => Expired
            # - Deadline <= 10 days => Critical
            # - Deadline <= 21 days => High
            # - Else Medium
            days_remaining = None
            try:
                # Use existing parser if available; our normalization emits numeric formats.
                dt = getattr(rfp_module, "_parse_deadline", None)
                if callable(dt):
                    parsed = dt(deadline_str)
                else:
                    parsed = None
                if parsed:
                    delta = parsed - datetime.now()
                    days_remaining = int(delta.total_seconds() // 86400)
            except Exception:
                days_remaining = None

            value_high = _value_exceeds_2cr(estimated_value or "")
            if isinstance(days_remaining, int):
                if days_remaining < 0:
                    return "Expired"
                if days_remaining <= 10:
                    return "Critical"
                if days_remaining <= 21:
                    return "High"
                return "Medium"
            return "High" if value_high else "Medium"

        def _extract_rfp_id(text: str) -> Optional[str]:
            for lab in ["Bid Reference No.", "Bid Reference No", "Bid Reference", "RFP Reference", "Tender Ref", "Tender Reference"]:
                v = _extract_after_label(text, lab)
                if v:
                    # trim trailing punctuation
                    return v.strip().strip(". ")
            return None

        def _extract_scope_items_count(text: str) -> Optional[int]:
            # Count bullets under "Scope Summary:" section
            m = re.search(r"(?is)scope\s*summary\s*:\s*(.+)$", text)
            if not m:
                return None
            tail = m.group(1)
            lines = [ln.strip() for ln in tail.splitlines()]
            count = 0
            for ln in lines:
                if not ln:
                    # stop once we hit a blank after bullets have started
                    if count > 0:
                        break
                    continue
                if re.match(r"^[-*•]\s+", ln):
                    count += 1
                    continue
                # stop when we reach another field label
                if re.match(r"^(submission\s+deadline|estimated\s+value|buyer\s*/\s*organization|buyer|organization)\s*:", ln, flags=re.IGNORECASE):
                    break
            return count if count > 0 else None

        def _infer_domain(text: str) -> str:
            t = (text or "").lower()

            # Default to Electrical if power/control cable family is mentioned.
            if any(k in t for k in [
                "power cable",
                "control cable",
                "xlpe",
                "ht ",
                " ht",
                "lt ",
                " lt",
                "11kv",
                "33kv",
                "66kv",
                "132kv",
                "cable",
                "armoured",
                "unarmoured",
            ]):
                return "Electrical"

            if any(k in t for k in [" epc ", "\nepc\n", "engineering procurement", "procurement construction", "turnkey", "lump sum"]):
                return "EPC"

            if any(k in t for k in ["paint", "painting", "coating", "epoxy", "polyurethane", "powder coating"]):
                return "Paints"

            if "electrical" in t:
                return "Electrical"
            return "General"

        def _extract_title(subject: str, body_text: str) -> str:
            s = (subject or "").strip()
            if s:
                return s
            # fallback: first non-empty paragraph
            for para in (body_text or "").split("\n\n"):
                p = para.strip()
                if p:
                    # cap length
                    return p[:140]
            return "Untitled RFP"

        def _gmail_email_to_rfp(email_obj: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
            try:
                message_id = str(email_obj.get("message_id") or "").strip()
                sender = str(email_obj.get("sender") or "").strip()
                subject = str(email_obj.get("subject") or "").strip()
                body_text = str(email_obj.get("body_text") or "").strip()
                received_ts = str(email_obj.get("received_timestamp") or "").strip()

                if not message_id or (not subject and not body_text):
                    return None

                rfp_id = _extract_rfp_id(body_text) or _stable_id_fallback(message_id, subject, body_text)
                rfp_title = _extract_title(subject, body_text)
                buyer = _extract_after_label(body_text, "Buyer / Organization") or _extract_after_label(body_text, "Buyer") or ""
                submission_deadline_raw = _extract_deadline_anywhere(body_text)
                submission_deadline = _normalize_deadline_for_priority(submission_deadline_raw) or (submission_deadline_raw or "")
                estimated_value_raw = _extract_estimated_value_anywhere(body_text)
                estimated_value = (estimated_value_raw or "").strip()
                if estimated_value:
                    estimated_value = re.sub(r"\s+", " ", estimated_value.replace("\u2013", "-").replace("\u2014", "-"))
                scope_items = _extract_scope_items_count(body_text)
                domain = _infer_domain(subject + "\n" + body_text)

                # Gmail-specific priority override (rule-based; no LLM)
                gmail_priority = _compute_gmail_priority(submission_deadline, estimated_value)

                raw = {
                    "rfp_id": rfp_id,
                    "rfp_title": rfp_title,
                    "buyer": buyer,
                    "submission_deadline": submission_deadline,
                    "estimated_project_value": estimated_value,
                    "scope_items": scope_items or 0,
                    "tender_source": sender,
                    "domain": domain,
                    "source": "gmail",
                    "gmail_message_id": message_id,
                    "gmail_received_timestamp": received_ts,
                }

                # Use the *existing* priority logic via the existing formatter
                formatted = rfp_module.format_sales_for_frontend(raw, index)
                formatted["source"] = "gmail"
                formatted["gmail_message_id"] = message_id
                formatted["gmail_received_timestamp"] = received_ts

                # Ensure UI shows Gmail-specific priority per requirements.
                formatted["priority"] = gmail_priority
                return formatted
            except Exception as e:
                print(f"[Gmail Parse] ⚠️ Skipping email due to parse error: {e}")
                return None

        gmail_emails = list_gmail_emails()
        gmail_rfps: List[Dict[str, Any]] = []
        gmail_pdf_paths: List[str] = []
        pdf_text_cache = state.get("pdf_text_cache", {}) or {}

        for idx, email_obj in enumerate(gmail_emails):
            rfp = _gmail_email_to_rfp(email_obj, 1000 + idx)
            if not rfp:
                continue

            # Create a stable, existing "pdf" placeholder so downstream agents
            # (Master/Technical) can operate without modifications.
            message_id = str(email_obj.get("message_id") or "").strip()
            subject = str(email_obj.get("subject") or "")
            body_text = str(email_obj.get("body_text") or "")
            digest = hashlib.md5((message_id + "|" + subject).encode("utf-8", errors="ignore")).hexdigest()[:12]
            dummy_pdf = f"gmail_{digest}.pdf"
            try:
                # Ensure file exists (content unused because we rely on cache)
                with open(dummy_pdf, "ab"):
                    pass
            except Exception as e:
                print(f"[Gmail Integration] ⚠️ Could not create placeholder PDF {dummy_pdf}: {e}")
                continue

            pdf_text_cache[dummy_pdf] = body_text
            gmail_rfps.append(rfp)
            gmail_pdf_paths.append(dummy_pdf)

        if gmail_rfps:
            sales_data = list(sales_data) + gmail_rfps
            state["sales_output"] = sales_data
            state["pdf_paths"] = list(state.get("pdf_paths", []) or []) + gmail_pdf_paths
            state["pdf_text_cache"] = pdf_text_cache

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


class GmailRfpEvent(BaseModel):
    # New schema (preferred)
    message_id: Optional[str] = None
    sender: str
    subject: str
    body_text: Optional[str] = ""
    received_timestamp: Optional[str] = None

    # Backward-compatible fields from older listener payloads
    timestamp: Optional[str] = None


@app.post("/integrations/gmail/rfp-event")
def ingest_gmail_rfp_event(event: GmailRfpEvent):
    # Store raw email (deduped) + create a notification. Do NOT trigger agents.
    received_ts = event.received_timestamp or event.timestamp or datetime.utcnow().isoformat() + "Z"
    body_text = event.body_text or ""

    message_id = (event.message_id or "").strip()
    if not message_id:
        basis = f"{event.sender}|{event.subject}|{received_ts}|{body_text}"
        message_id = "legacy-" + hashlib.md5(basis.encode("utf-8", errors="ignore")).hexdigest()[:16]

    raw_email = {
        "message_id": message_id,
        "sender": event.sender,
        "subject": event.subject,
        "body_text": body_text,
        "received_timestamp": received_ts,
    }

    inserted = add_gmail_email(raw_email)
    if inserted:
        # Minimal notification schema for UI
        notif = {
            "id": str(uuid.uuid4()),
            "type": "discovery",
            "title": f"New Gmail RFP: {event.subject}" if event.subject else "New Gmail RFP received",
            "description": f"From: {event.sender}",
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
