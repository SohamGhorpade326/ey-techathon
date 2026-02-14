from typing import List, Dict, Any
import threading

# Thread-safe lock
_lock = threading.Lock()

# Global in-memory store
_STORE = {
    "sales": [],
    "pdf_paths": [],  # Preserve PDF paths for skip mode
    "pdf_text_cache": {},  # Preserve extracted text for skip mode
    "master": {},
    "technical": {},
    "pricing": {},
    "compliance": [],
    "proposal": "",
    # Gmail ingestion (kept across pipeline runs)
    "gmail_emails": [],
    "notifications": [],
    # Strategic Skip infrastructure
    "skipped_rfp_ids": []
}


def add_gmail_email(email_obj: Dict[str, Any]) -> bool:
    """Add a Gmail email raw object (deduped by message_id). Returns True if inserted."""
    message_id = str(email_obj.get("message_id") or "").strip()
    if not message_id:
        return False

    with _lock:
        existing = _STORE.get("gmail_emails", [])
        if any(str(e.get("message_id")) == message_id for e in existing):
            return False
        existing.append(email_obj)
        _STORE["gmail_emails"] = existing
        return True


def list_gmail_emails() -> List[Dict[str, Any]]:
    with _lock:
        return list(_STORE.get("gmail_emails", []))


def add_notification(notification: Dict[str, Any]) -> bool:
    """Add a notification (deduped by message_id if present). Returns True if inserted."""
    with _lock:
        existing = _STORE.get("notifications", [])
        message_id = notification.get("message_id")
        
        # Deduplicate by message_id for Gmail notifications
        if message_id:
            if any(n.get("message_id") == message_id for n in existing):
                return False
        
        existing.append(notification)
        _STORE["notifications"] = existing
        return True


def list_notifications() -> List[Dict[str, Any]]:
    with _lock:
        return list(_STORE.get("notifications", []))


# Track which Gmail emails have been processed by sales agent
def mark_gmail_email_processed(message_id: str) -> None:
    """Mark a Gmail email as processed by sales agent"""
    with _lock:
        processed = _STORE.get("processed_gmail_ids", set())
        processed.add(message_id)
        _STORE["processed_gmail_ids"] = processed

def is_gmail_email_processed(message_id: str) -> bool:
    """Check if a Gmail email has already been processed"""
    with _lock:
        processed = _STORE.get("processed_gmail_ids", set())
        return message_id in processed

def get_unprocessed_gmail_emails() -> List[Dict[str, Any]]:
    """Get Gmail emails from the last 10 minutes (time-based, not processed flag)"""
    from datetime import datetime, timedelta, timezone
    
    with _lock:
        all_emails = _STORE.get("gmail_emails", [])
        print(f"[Gmail Store] Total Gmail emails in store: {len(all_emails)}")
        
        # Use timezone-aware datetime for proper comparison
        now = datetime.now(timezone.utc)
        ten_minutes_ago = now - timedelta(minutes=10)
        
        recent_emails = []
        for e in all_emails:
            # Parse received_timestamp
            timestamp_str = e.get("received_timestamp")
            if not timestamp_str:
                print(f"[Gmail Store] Skipping email with no timestamp: {e.get('subject', 'Unknown')}")
                continue
            try:
                # Parse ISO timestamp (handles both 'Z' and '+00:00' formats)
                email_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                # Make sure email_time is timezone-aware
                if email_time.tzinfo is None:
                    email_time = email_time.replace(tzinfo=timezone.utc)
                
                age_minutes = (now - email_time).total_seconds() / 60
                print(f"[Gmail Store] Email '{e.get('subject', 'Unknown')[:50]}' age: {age_minutes:.1f} min")
                
                if email_time >= ten_minutes_ago:
                    recent_emails.append(e)
                    print(f"[Gmail Store] ✓ Including email (within 10 min window)")
                else:
                    print(f"[Gmail Store] ✗ Skipping email (older than 10 min)")
            except Exception as ex:
                print(f"[Gmail Store] Error parsing timestamp '{timestamp_str}': {ex}")
                continue
        
        print(f"[Gmail Store] Returning {len(recent_emails)} recent emails (last 10 min)")
        return recent_emails

def clear_notifications() -> None:
    """Clear all notifications"""
    with _lock:
        _STORE["notifications"] = []

def clear_gmail_data() -> None:
    """Clear all Gmail emails, processed IDs, and notifications"""
    with _lock:
        _STORE["gmail_emails"] = []
        _STORE["processed_gmail_ids"] = set()
        _STORE["notifications"] = []

# SALES
def save_sales_output(data: List[Dict[str, Any]]) -> None:
    with _lock:
        _STORE["sales"] = data if data else []

def get_sales_output() -> List[Dict[str, Any]]:
    with _lock:
        return _STORE.get("sales", [])

# PDF PATHS AND TEXT CACHE (for skip mode)
def save_pdf_paths(paths: List[str]) -> None:
    with _lock:
        _STORE["pdf_paths"] = paths if paths else []

def get_pdf_paths() -> List[str]:
    with _lock:
        return _STORE.get("pdf_paths", [])

def save_pdf_text_cache(cache: Dict[str, str]) -> None:
    with _lock:
        _STORE["pdf_text_cache"] = cache if cache else {}

def get_pdf_text_cache() -> Dict[str, str]:
    with _lock:
        return _STORE.get("pdf_text_cache", {})

# MASTER
def save_master_output(data: Dict[str, Any]) -> None:
    with _lock:
        _STORE["master"] = data if data else {}

def get_master_output() -> Dict[str, Any]:
    with _lock:
        return _STORE.get("master", {})

# TECHNICAL
def save_technical_output(data: Dict[str, Any]) -> None:
    with _lock:
        _STORE["technical"] = data if data else {}

def get_technical_output() -> Dict[str, Any]:
    with _lock:
        return _STORE.get("technical", {})

# PRICING
def save_pricing_output(data: Dict[str, Any]) -> None:
    with _lock:
        _STORE["pricing"] = data if data else {}

def get_pricing_output() -> Dict[str, Any]:
    with _lock:
        return _STORE.get("pricing", {})

# COMPLIANCE TABLE
def save_compliance_output(data: List[Dict[str, Any]]) -> None:
    with _lock:
        _STORE["compliance"] = data if data else []

def get_compliance_output() -> List[Dict[str, Any]]:
    with _lock:
        return _STORE.get("compliance", [])

# PROPOSAL
def save_proposal_output(data: str) -> None:
    with _lock:
        _STORE["proposal"] = data if data else ""

def get_proposal_output() -> str:
    with _lock:
        return _STORE.get("proposal", "")

# CLEAR ALL
def clear_all() -> None:
    with _lock:
        _STORE["sales"] = []
        _STORE["pdf_paths"] = []
        _STORE["pdf_text_cache"] = {}
        _STORE["master"] = {}
        _STORE["technical"] = {}
        _STORE["pricing"] = {}
        _STORE["compliance"] = []
        _STORE["proposal"] = ""
        # Note: Gmail emails, processed IDs, notifications, and skipped_rfp_ids persist
        # skipped_rfp_ids is cleared separately in normal mode but persists in skip mode

# SKIPPED RFP IDS MANAGEMENT
def add_skipped_rfp_id(rfp_id: str) -> None:
    """Add an RFP ID to the skipped list"""
    with _lock:
        if rfp_id and rfp_id not in _STORE["skipped_rfp_ids"]:
            _STORE["skipped_rfp_ids"].append(rfp_id)
            print(f"[Store] Added {rfp_id} to skipped list")

def get_skipped_rfp_ids() -> List[str]:
    """Get the list of skipped RFP IDs"""
    with _lock:
        return list(_STORE.get("skipped_rfp_ids", []))

def clear_skipped_rfp_ids() -> None:
    """Clear the skipped RFP IDs list"""
    with _lock:
        _STORE["skipped_rfp_ids"] = []
