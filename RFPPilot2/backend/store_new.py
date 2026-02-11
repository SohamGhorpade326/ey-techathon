from typing import List, Dict, Any
import threading

# Thread-safe lock
_lock = threading.Lock()

# Global in-memory store
_STORE = {
    "sales": [],
    "master": {},
    "technical": {},
    "pricing": {},
    "compliance": [],
    "proposal": "",
    # Gmail ingestion (kept across pipeline runs)
    "gmail_emails": [],
    "notifications": []
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


def add_notification(notification: Dict[str, Any]) -> None:
    with _lock:
        existing = _STORE.get("notifications", [])
        existing.append(notification)
        _STORE["notifications"] = existing


def list_notifications() -> List[Dict[str, Any]]:
    with _lock:
        return list(_STORE.get("notifications", []))

# SALES
def save_sales_output(data: List[Dict[str, Any]]) -> None:
    with _lock:
        _STORE["sales"] = data if data else []

def get_sales_output() -> List[Dict[str, Any]]:
    with _lock:
        return _STORE.get("sales", [])

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
        _STORE["master"] = {}
        _STORE["technical"] = {}
        _STORE["pricing"] = {}
        _STORE["compliance"] = []
        _STORE["proposal"] = ""
