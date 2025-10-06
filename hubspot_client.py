"""HubSpot HTTP helpers — logic copied from original app.py."""
from config import *
import requests
import pandas as pd
import streamlit as st
import os

# ---- hs_headers ----

def hs_headers() -> dict:
    return {"Authorization": f"Bearer {HUBSPOT_TOKEN}"}

def _hs_token() -> str:
    """
    Resolve a HubSpot private app token from config or environment.
    We try a few common names so you do not have to touch code:
      - HUBSPOT_PRIVATE_APP_TOKEN
      - HUBSPOT_API_KEY
      - HUBSPOT_TOKEN
      - HS_TOKEN
    Put one of these in Streamlit Secrets.
    """
    for key in ("HUBSPOT_PRIVATE_APP_TOKEN", "HUBSPOT_API_KEY", "HUBSPOT_TOKEN", "HS_TOKEN"):
        val = globals().get(key) or os.getenv(key)
        if val:
            return val
    raise RuntimeError(
        "HubSpot token not configured. Set HUBSPOT_PRIVATE_APP_TOKEN (or HUBSPOT_API_KEY) "
        "in Streamlit Cloud → Settings → Secrets."
    )

def _hs_headers() -> dict:
    """Standard JSON + Bearer auth headers for HubSpot HTTP calls."""
    return {
        "Authorization": f"Bearer {_hs_token()}",
        "Content-Type": "application/json",
    }

def _hs_get(path: str, params: dict | None = None) -> dict:
    """Low-level GET wrapper for HubSpot."""
    base = "https://api.hubapi.com"
    url = f"{base}{path}"
    r = requests.get(url, headers=_hs_headers(), params=params or {}, timeout=60)
    r.raise_for_status()
    return r.json()

def _hs_post(path: str, payload: dict) -> dict:
    """Low-level POST wrapper for HubSpot."""
    base = "https://api.hubapi.com"
    url = f"{base}{path}"
    r = requests.post(url, headers=_hs_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()

def _hs_patch(path: str, payload: dict) -> dict:
    """Low-level PATCH wrapper for HubSpot."""
    base = "https://api.hubapi.com"
    url = f"{base}{path}"
    r = requests.patch(url, headers=_hs_headers(), json=payload, timeout=60)
    r.raise_for_status()
    return r.json()
# ---- hs_get_owner_info ----

def hs_get_owner_info(owner_id):
    """Get owner information by ID"""
    try:
        url = f"{HS_ROOT}/crm/v3/owners/{owner_id}"
        response = requests.get(url, headers=hs_headers(), timeout=10)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None

def _search_once(
    payload: dict,
    *,
    total_cap: int = 1000,
    endpoint: str = "/crm/v3/objects/deals/search"
) -> pd.DataFrame:
    """
    Execute a HubSpot CRM search with safe pagination and return a flat DataFrame
    of 'properties' plus 'id'. This mirrors the monolith's private helper.
    """
    out = []
    after = None
    fetched = 0
    while True:
        body = dict(payload)  # shallow copy so we do not mutate the caller's payload
        # Respect a caller-provided 'limit' but never exceed total_cap
        limit = int(body.get("limit", 100))
        limit = min(limit, max(0, total_cap - fetched))
        if limit <= 0:
            break
        body["limit"] = limit
        if after is not None:
            body["after"] = after

        j = _hs_post(endpoint, body)
        results = j.get("results", [])
        out.extend(results)
        fetched += len(results)

        after = j.get("paging", {}).get("next", {}).get("after")
        if not after or fetched >= total_cap:
            break

    # Flatten to DataFrame (properties + id)
    rows = []
    for r in out:
        props = r.get("properties", {}) or {}
        props["id"] = r.get("id")
        rows.append(props)
    return pd.DataFrame(rows)

# ---- hs_get_deal_property_options ----

def hs_get_deal_property_options(property_name: str) -> list[dict]:
    fallback_states = [{"label": s, "value": s} for s in ["VIC","NSW","QLD","SA","WA","TAS","NT","ACT"]]
    try:
        url = f"{HS_PROP_URL}/{property_name}"
        r = requests.get(url, headers=hs_headers(), params={"archived": "false"}, timeout=8)
        r.raise_for_status()
        data = r.json()
        options = data.get("options", []) or []
        out = []
        for opt in options:
            value = str(opt.get("value") or "").strip()
            label = str(opt.get("label") or opt.get("displayValue") or value).strip()
            if value:
                out.append({"label": label or value, "value": value})
        return out or fallback_states
    except requests.exceptions.RequestException:
        st.info("Network issue while fetching state options. Using default states.")
        return fallback_states
    except Exception:
        st.info("Unexpected issue while fetching state options. Using default states.")
        return fallback_states



# ---- hs_search_deals_by_date_property ----

def hs_search_deals_by_date_property(*,
    pipeline_id: str, stage_id: str, state_value: str,
    date_property: str, date_eq_ms: int | None,
    date_start_ms: int | None, date_end_ms: int | None,
    total_cap: int = HS_TOTAL_CAP
) -> pd.DataFrame:
    filters = [
        {"propertyName": "pipeline", "operator": "EQ", "value": pipeline_id},
        {"propertyName": "dealstage", "operator": "EQ", "value": stage_id},
        {"propertyName": "car_location_at_time_of_sale", "operator": "EQ", "value": state_value},
    ]
    if date_eq_ms is not None:
        filters.append({"propertyName": date_property, "operator": "EQ", "value": int(date_eq_ms)})
    else:
        if date_start_ms is not None:
            filters.append({"propertyName": date_property, "operator": "GTE", "value": int(date_start_ms)})
        if date_end_ms is not None:
            filters.append({"propertyName": date_property, "operator": "LTE", "value": int(date_end_ms)})
    payload = {"filterGroups": [{"filters": filters}], "properties": DEAL_PROPS, "limit": HS_PAGE_LIMIT}
    df = _search_once(payload, total_cap=total_cap)
    if df.empty and date_eq_ms is not None:
        widen = 12 * 3600 * 1000
        filters[-1] = {"propertyName": date_property, "operator": "GTE", "value": int(date_eq_ms - widen)}
        filters.append({"propertyName": date_property, "operator": "LTE", "value": int(date_eq_ms + widen)})
        payload = {"filterGroups": [{"filters": filters}], "properties": DEAL_PROPS, "limit": HS_PAGE_LIMIT}
        df = _search_once(payload, total_cap=total_cap)
    return df



# ---- hs_search_deals_by_appointment_and_stages ----

def hs_search_deals_by_appointment_and_stages(appointment_id: str, pipeline_id: str, stage_ids: set[str]) -> pd.DataFrame:
    filters = [
        {"propertyName": "pipeline", "operator": "EQ", "value": pipeline_id},
        {"propertyName": "appointment_id", "operator": "EQ", "value": str(appointment_id).strip()},
        {"propertyName": "dealstage", "operator": "IN", "values": list(stage_ids)},
    ]
    payload = {"filterGroups": [{"filters": filters}], "properties": DEAL_PROPS, "limit": HS_PAGE_LIMIT}
    return _search_once(payload, total_cap=HS_TOTAL_CAP)



# ---- hs_deals_to_contacts_map ----

def hs_deals_to_contacts_map(deal_ids: list[str]) -> dict[str, list[str]]:
    out = {str(d): [] for d in deal_ids}
    if not deal_ids: return out
    url = f"{HS_ROOT}/crm/v4/objects/deals/batch/read"
    payload = {"properties": [], "inputs": [{"id": str(d)} for d in deal_ids], "associations": ["contacts"]}
    try:
        r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
        r.raise_for_status()
        for item in r.json().get("results", []):
            did = str(item.get("id"))
            contacts = [a.get("id") for a in item.get("associations", {}).get("contacts", [])]
            out[did] = [str(x) for x in contacts if x]
    except Exception as e:
        st.warning(f"Could not read deal→contacts associations: {e}")
    return out



# ---- hs_contacts_to_deals_map ----

def hs_contacts_to_deals_map(contact_ids: list[str]) -> dict[str, list[str]]:
    out = {str(c): [] for c in contact_ids}
    if not contact_ids: return out
    url = f"{HS_ROOT}/crm/v4/objects/contacts/batch/read"
    payload = {"properties": [], "inputs": [{"id": str(c)} for c in contact_ids], "associations": ["deals"]}
    try:
        r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
        r.raise_for_status()
        for item in r.json().get("results", []):
            cid = str(item.get("id"))
            deals = [a.get("id") for a in item.get("associations", {}).get("deals", [])]
            out[cid] = [str(x) for x in deals if x]
    except Exception as e:
        st.warning(f"Could not read contact→deals associations: {e}")
    return out



# ---- hs_batch_read_deals ----

def hs_batch_read_deals(deal_ids: list[str], props: list[str]) -> dict[str, dict]:
    out = {}
    if not deal_ids: return out
    url = f"{HS_ROOT}/crm/v3/objects/deals/batch/read"
    for i in range(0, len(deal_ids), 100):
        chunk = deal_ids[i:i+100]
        payload = {"properties": props, "inputs": [{"id": str(d)} for d in chunk]}
        try:
            r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
            r.raise_for_status()
            for item in r.json().get("results", []):
                out[str(item.get("id"))] = item.get("properties", {}) or {}
        except Exception as e:
            st.warning(f"Could not batch read deals (props={props}): {e}")
    return out

# ============ Aircall ============



# ---- update_deals_sms_sent ----

def update_deals_sms_sent(deal_ids: list[str]) -> tuple[int, int]:
    """
    Update the td_reminder_sms_sent property to 'true' for the given deal IDs.
    Returns (success_count, failure_count)
    """
    if not deal_ids:
        return 0, 0
    
    success_count = 0
    failure_count = 0
    
    url = f"{HS_ROOT}/crm/v3/objects/deals/batch/update"
    
    # Process in batches of 100 (HubSpot limit)
    for i in range(0, len(deal_ids), 100):
        batch = deal_ids[i:i+100]
        
        inputs = []
        for deal_id in batch:
            inputs.append({
                "id": str(deal_id),
                "properties": {
                    "td_reminder_sms_sent": "true"  # CHANGED FROM "Yes" to "true"
                }
            })
        
        payload = {"inputs": inputs}
        
        try:
            response = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
            if response.status_code == 200:
                success_count += len(batch)
            else:
                failure_count += len(batch)
                st.warning(f"Failed to update batch: {response.text[:200]}")
        except Exception as e:
            failure_count += len(batch)
            st.warning(f"Error updating deals: {str(e)}")
    
    return success_count, failure_count



# ---- export_sms_update_list ----

def export_sms_update_list(phone_to_deals: dict, sent_phones: list) -> pd.DataFrame:
    """
    Create a DataFrame with deal IDs that need to be updated after SMS send.
    """
    update_records = []
    for phone in sent_phones:
        if phone in phone_to_deals:
            for deal_id in phone_to_deals[phone]:
                update_records.append({
                    "Deal ID": deal_id,
                    "td_reminder_sms_sent": "true",
                    "Phone": phone,
                    "Update Time": datetime.now(MEL_TZ).strftime("%Y-%m-%d %H:%M:%S")
                })
    
    return pd.DataFrame(update_records)



# ---- get_contact_ids_for_deal ----

def get_contact_ids_for_deal(deal_id):
    """Get contact IDs associated with a deal"""
    headers = {"Authorization": f"Bearer {HUBSPOT_TOKEN}"}  
    try:
        url = f"{HS_ROOT}/crm/v3/objects/deals/{deal_id}/associations/contacts"
        response = requests.get(url, headers=headers, timeout=25)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            return [result.get("toObjectId") or result.get("id") for result in results]
        else:
            return []
    except Exception:
        return []



# ---- get_consolidated_notes_for_deal ----

def get_consolidated_notes_for_deal(deal_id):
    """Get all consolidated notes for a deal"""
    headers = {"Authorization": f"Bearer {HUBSPOT_TOKEN}"}
    
    contact_ids = get_contact_ids_for_deal(deal_id)
    
    if not contact_ids:
        return "No notes"
    
    all_formatted_notes = []
    
    for contact_id in contact_ids:
        note_ids = get_contact_note_ids(contact_id)
        
        if note_ids:
            notes = get_notes_content(note_ids)
            
            for note in notes:
                props = note.get("properties", {})
                body = props.get("hs_note_body", "")
                timestamp = props.get("hs_timestamp") or props.get("hs_createdate", "")
                owner_id = props.get("hubspot_owner_id")
                
                if body and body.strip():
                    # Clean HTML from body
                    import re
                    clean_body = re.sub(r'<[^>]+>', '', body).strip()
                    clean_body = clean_body.replace('&nbsp;', ' ').replace('&amp;', '&')
                    
                    if clean_body:
                        # Format timestamp
                        date_str = "Unknown Date"
                        if timestamp:
                            try:
                                from datetime import datetime
                                if len(str(timestamp)) > 10:  # milliseconds
                                    dt = datetime.fromtimestamp(int(timestamp) / 1000)
                                else:  # seconds
                                    dt = datetime.fromtimestamp(int(timestamp))
                                date_str = dt.strftime("%Y-%m-%d %H:%M")
                            except:
                                date_str = str(timestamp)
                        
                        # Get owner name
                        owner_name = get_owner_name(owner_id)
                        
                        formatted_note = f"[{date_str}] ({owner_name}) {clean_body}"
                        all_formatted_notes.append(formatted_note)
    
    if all_formatted_notes:
        return "\n\n".join(all_formatted_notes)
    else:
        return "No notes"



# ---- get_deals_by_owner_and_daterange ----

def get_deals_by_owner_and_daterange(start_date, end_date, state_val, selected_owners):
    """Simplified version - get deals by date range and state only"""
    try:
        start_ms, _ = mel_day_bounds_to_epoch_ms(start_date)
        _, end_ms = mel_day_bounds_to_epoch_ms(end_date)
        
        raw_deals = hs_search_deals_by_date_property(
            pipeline_id=PIPELINE_ID,
            stage_id="1119198253",
            state_value=state_val,
            date_property="td_conducted_date",
            date_eq_ms=None,
            date_start_ms=start_ms,
            date_end_ms=end_ms,
            total_cap=HS_TOTAL_CAP
        )
        
        if raw_deals is None or (isinstance(raw_deals, list) and len(raw_deals) == 0) or (isinstance(raw_deals, pd.DataFrame) and raw_deals.empty):
            return pd.DataFrame()
            
        return prepare_deals(raw_deals)
        
    except Exception as e:
        st.error(f"Error fetching deals: {str(e)}")
        return pd.DataFrame()


# ============ NEW: Appointment ID based car filtering ============



# ---- get_deals_by_appointment_id ----

def get_deals_by_appointment_id(appointment_id: str) -> list[str]:
    """Get all deal IDs that have the given appointment_id"""
    if not appointment_id:
        return []
    
    try:
        url = f"{HS_ROOT}/crm/v3/objects/deals/search"
        payload = {
            "filterGroups": [
                {
                    "filters": [
                        {
                            "propertyName": "appointment_id",
                            "operator": "EQ",
                            "value": str(appointment_id).strip()
                        }
                    ]
                }
            ],
            "properties": ["hs_object_id", "dealstage", "appointment_id"],
            "limit": 100
        }
        response = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
        
        if response.status_code == 200:
            data = response.json()
            deals = data.get("results", [])
            return [deal["properties"]["hs_object_id"] for deal in deals]
        else:
            st.warning(f"Error searching deals by appointment_id: {response.text}")
            return []
            
    except Exception as e:
        st.warning(f"Exception searching deals by appointment_id: {e}")
        return []



# ---- hs_deals_to_contacts_map ----

def hs_deals_to_contacts_map(deal_ids: list[str]) -> dict[str, list[str]]:
    out = {str(d): [] for d in deal_ids}
    if not deal_ids: return out
    url = f"{HS_ROOT}/crm/v4/objects/deals/batch/read"
    payload = {"properties": [], "inputs": [{"id": str(d)} for d in deal_ids], "associations": ["contacts"]}
    try:
        r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
        r.raise_for_status()
        for item in r.json().get("results", []):
            did = str(item.get("id"))
            contacts = [a.get("id") for a in item.get("associations", {}).get("contacts", [])]
            out[did] = [str(x) for x in contacts if x]
    except Exception as e:
        st.warning(f"Could not read deal→contacts associations: {e}")
    return out



# ---- hs_contacts_to_deals_map ----

def hs_contacts_to_deals_map(contact_ids: list[str]) -> dict[str, list[str]]:
    out = {str(c): [] for c in contact_ids}
    if not contact_ids: return out
    url = f"{HS_ROOT}/crm/v4/objects/contacts/batch/read"
    payload = {"properties": [], "inputs": [{"id": str(c)} for c in contact_ids], "associations": ["deals"]}
    try:
        r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
        r.raise_for_status()
        for item in r.json().get("results", []):
            cid = str(item.get("id"))
            deals = [a.get("id") for a in item.get("associations", {}).get("deals", [])]
            out[cid] = [str(x) for x in deals if x]
    except Exception as e:
        st.warning(f"Could not read contact→deals associations: {e}")
    return out



# ---- hs_batch_read_deals ----

def hs_batch_read_deals(deal_ids: list[str], props: list[str]) -> dict[str, dict]:
    out = {}
    if not deal_ids: return out
    url = f"{HS_ROOT}/crm/v3/objects/deals/batch/read"
    for i in range(0, len(deal_ids), 100):
        chunk = deal_ids[i:i+100]
        payload = {"properties": props, "inputs": [{"id": str(d)} for d in chunk]}
        try:
            r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
            r.raise_for_status()
            for item in r.json().get("results", []):
                out[str(item.get("id"))] = item.get("properties", {}) or {}
        except Exception as e:
            st.warning(f"Could not batch read deals (props={props}): {e}")
    return out

# ============ Aircall ============


