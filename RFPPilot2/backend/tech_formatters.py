def format_technical_for_frontend(raw_output: dict) -> dict:
    """
    Formats technical agent raw output for frontend display.
    Converts internal technical data to UI-friendly format.
    """
    if not raw_output or not isinstance(raw_output, dict):
        return {
            "rfp_id": "Unknown",
            "status": "Error",
            "items": [],
            "summary": {
                "total_items": 0,
                "matched_items": 0,
                "warning_items": 0,
                "unmatched_items": 0
            }
        }
    
    items = raw_output.get("items", [])
    rfp_id = raw_output.get("rfp_id", "Unknown")
    domain = raw_output.get("domain", "General")
    
    # Process items for frontend
    frontend_items = []
    matched_count = 0
    warning_count = 0
    unmatched_count = 0
    
    for item in items:
        status = item.get("status", "Unknown")
        match_percent = item.get("spec_match_percent", 0)
        
        # Count status types
        if status == "Matched":
            matched_count += 1
        elif status == "Warning":
            warning_count += 1
        elif status in ["Review", "No Match"]:
            unmatched_count += 1
            
        frontend_item = {
            "item_name": item.get("item_name", "Unknown Item"),
            "required_specs_text": item.get("required_technical_specs", "Refer RFP"),
            "best_match_sku": item.get("best_match_sku", "NO_MATCH"),
            "spec_match_percent": round(match_percent, 1),
            "status": status,
            "top_3_alternatives": item.get("top_3_skus", [])[:3]
        }
        frontend_items.append(frontend_item)
    
    return {
        "rfp_id": rfp_id,
        "domain": domain,
        "status": "Completed",
        "items": frontend_items,
        "summary": {
            "total_items": len(items),
            "matched_items": matched_count,
            "warning_items": warning_count,
            "unmatched_items": unmatched_count,
            "success_rate": round((matched_count / len(items)) * 100, 1) if items else 0
        }
    }


def build_comparison_matrix(item: dict, domain: str = "General") -> dict:
    """
    Builds comparison table dynamically based on domain.
    Much more realistic for Civil vs Electrical tenders.
    """

    domain = (domain or "").lower()

    # -------------------------
    # Domain-aware spec fields
    # -------------------------
    if "rf" in domain or "coax" in domain or "assembly" in domain or "frequency" in domain or "vswr" in domain:
        spec_keys = [
            "Frequency Range",
            "Impedance (Ω)",
            "Connector Type",
            "Length (m)",
            "VSWR",
            "Insertion Loss (dB/m)"
        ]

    elif "electrical" in domain:
        spec_keys = [
            "Conductor",
            "Size (sqmm)",
            "Voltage (kV)",
            "Insulation",
            "Armouring",
            "Sheath"
        ]

    elif "civil" in domain:
        spec_keys = [
            "Material Type",
            "Coverage (sqft/L)",
            "Thickness",
            "Application Method",
            "Drying Time",
            "Standards"
        ]

    else:
        spec_keys = [
            "Material",
            "Specification",
            "Standards",
            "Warranty"
        ]

    rows = []

    for key in spec_keys:
        row = {
            "parameter": key,
            "rfp_requirement": "Refer RFP",
            "sku_1": None,
            "sku_2": None,
            "sku_3": None
        }

        for idx, sku in enumerate(item.get("top_3_skus", [])):
            row[f"sku_{idx+1}"] = {
                "sku_code": sku["sku_code"],
                "match": idx == 0
            }

        rows.append(row)

    return {
        "item_name": item["item_name"],
        "comparison_rows": rows
    }
