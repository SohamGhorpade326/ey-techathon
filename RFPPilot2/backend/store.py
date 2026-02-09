from typing import List, Dict, Any
from pymongo import MongoClient

try:
    client = MongoClient(
        "mongodb://localhost:27017",
        serverSelectionTimeoutMS=3000
    )
    client.server_info()

    db = client["agentic_rfp"]
    rfp_collection = db["rfps"]
    USE_MONGO = True
    print("[STORE] MongoDB connected")

except Exception:
    USE_MONGO = False
    _rfp_store: Dict[str, Dict[str, Any]] = {}
    _sales_output: List[Dict[str, Any]] = []  # Store sales agent output separately
    _master_output: Dict[str, Any] = {}  # Store master agent output separately
    print("[STORE] Using in-memory store")


def save_rfp(rfp: Dict[str, Any]) -> None:
    rfp_id = rfp.get("rfp_id")
    if not rfp_id:
        return

    if USE_MONGO:
        rfp_collection.update_one(
            {"rfp_id": rfp_id},
            {"$set": rfp},
            upsert=True
        )
    else:
        _rfp_store[rfp_id] = rfp


def get_all_rfps() -> List[Dict[str, Any]]:
    if USE_MONGO:
        return list(rfp_collection.find({}, {"_id": 0}))
    return list(_rfp_store.values())


def get_rfp(rfp_id: str):
    if USE_MONGO:
        return rfp_collection.find_one(
            {"rfp_id": rfp_id},
            {"_id": 0}
        )
    return _rfp_store.get(rfp_id)   # ✅ THIS WAS MISSING


def delete_all_rfps():
    global _sales_output, _master_output
    if USE_MONGO:
        rfp_collection.delete_many({})
    else:
        _rfp_store.clear()
        _sales_output = []
        _master_output = {}


def save_sales_output(sales_data: List[Dict[str, Any]]) -> None:
    """Save sales agent output (all extracted RFPs)"""
    global _sales_output
    if USE_MONGO:
        # Store in a separate collection for sales output
        sales_collection = db["sales_output"]
        sales_collection.delete_many({})  # Clear previous
        if sales_data:
            sales_collection.insert_many(sales_data)
    else:
        _sales_output = sales_data


def get_sales_output() -> List[Dict[str, Any]]:
    """Get sales agent output (all extracted RFPs)"""
    if USE_MONGO:
        sales_collection = db["sales_output"]
        return list(sales_collection.find({}, {"_id": 0}))
    return _sales_output


def save_master_output(master_data: Dict[str, Any]) -> None:
    """Save master agent output (technical + pricing summaries)"""
    global _master_output
    if USE_MONGO:
        master_collection = db["master_output"]
        master_collection.delete_many({})  # Clear previous
        master_collection.insert_one(master_data)
    else:
        _master_output = master_data


def get_master_output() -> Dict[str, Any]:
    """Get master agent output (technical + pricing summaries)"""
    if USE_MONGO:
        master_collection = db["master_output"]
        result = master_collection.find_one({}, {"_id": 0})
        return result if result else {}
    return _master_output
