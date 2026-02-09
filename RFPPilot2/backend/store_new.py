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
    "proposal": ""
}

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
