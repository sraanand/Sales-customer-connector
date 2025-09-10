# app.py
# Pawan Customer Connector — HubSpot + Aircall (Streamlit)
# Fixes:
#  - Message checkboxes default UNCHECKED
#  - Persist data/results in session_state so editing checkboxes doesn't reset the workflow
#  - Keep using td_booking_slot_time for "Time" in reminders
#  - Dedup table shows exact date/time; preview shows relative, as before

import os
import time
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# Get deployment timestamp - this will be set when the app starts
DEPLOYMENT_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

# Then update your header() function:
def header():
    cols = st.columns([1, 6, 1.2])
    with cols[0]:
        logo_path = next((p for p in ["H2.svg", "cars24_logo.svg", "logo.svg", "cars24.png"] if os.path.exists(p)), None)
        if logo_path: st.image(logo_path, width=100, use_container_width=False)
        else:
            st.markdown(
                f"<div style='height:40px;display:flex;align-items:center;'><div style='background:{PRIMARY};padding:6px 10px;border-radius:6px;'><span style='font-weight:800;color:#FFFFFF'>CARS24</span></div></div>",
                unsafe_allow_html=True
            )
    with cols[1]:
        st.markdown('<h1 class="header-title" style="margin:0;">Pawan Customer Connector</h1>', unsafe_allow_html=True)
    with cols[2]:
        if st.session_state.get("view","home")!="home":
            if st.button("← Back", key="back_btn", use_container_width=True):
                st.session_state["view"]="home"
        
        # Add timestamp in the same column as the back button
        st.caption(f"🔄 Deployed: {DEPLOYMENT_TIME}")
    
    st.markdown('<hr class="div"/>', unsafe_allow_html=True)

# ============ Keys / Setup ============
load_dotenv()
HUBSPOT_TOKEN     = os.getenv("HUBSPOT_TOKEN") or st.secrets.get("HUBSPOT_TOKEN", "")
AIRCALL_ID        = os.getenv("AIRCALL_ID") or st.secrets.get("AIRCALL_ID", "")
AIRCALL_TOKEN     = os.getenv("AIRCALL_TOKEN") or st.secrets.get("AIRCALL_TOKEN", "")
AIRCALL_NUMBER_ID = os.getenv("AIRCALL_NUMBER_ID") or st.secrets.get("AIRCALL_NUMBER_ID", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

# OpenAI client (support both new & legacy SDKs)
_openai_ok = False
try:
    import openai
    try:
        openai.api_key = OPENAI_API_KEY
        _openai_ok = True
    except Exception:
        _openai_ok = False
except Exception:
    openai = None
    _openai_ok = False

PREFERRED_MODELS = ["gpt-4o-mini", "o4-mini", "gpt-4o", "gpt-3.5-turbo"]

MEL_TZ = ZoneInfo("Australia/Melbourne")
UTC_TZ = timezone.utc

# HubSpot endpoints/props
HS_ROOT       = "https://api.hubspot.com"
HS_SEARCH_URL = f"{HS_ROOT}/crm/v3/objects/deals/search"
HS_PROP_URL   = f"{HS_ROOT}/crm/v3/properties/deals"
HS_PAGE_LIMIT = 100
HS_TOTAL_CAP  = 1000

# Aircall
AIRCALL_BASE_URL = "https://api.aircall.io/v1"

# Pipeline & stages (from you)
PIPELINE_ID        = "2345821"
STAGE_ENQUIRY_ID   = "1119198251"  # Enquiry (no TD)
STAGE_BOOKED_ID    = "1119198252"  # 2. TD Booked
STAGE_CONDUCTED_ID = "1119198253"  # 3. TD Conducted (no deposit)

OLD_LEAD_START_STAGES = {STAGE_ENQUIRY_ID, STAGE_BOOKED_ID, STAGE_CONDUCTED_ID}

# Active purchase stages (exclude old-lead follow-ups if any contact has these on another deal)
ACTIVE_PURCHASE_STAGE_IDS = {
    "8082239", "8082240", "8082241", "8082242", "8082243", "8406593",
    "14816089", "14804235", "14804236", "14804237", "14804238",
    "14804239", "14804240", "14804241"
}

DEAL_PROPS = [
    "hs_object_id", "dealname", "pipeline", "dealstage",
    "full_name", "email", "mobile", "phone",
    "appointment_id",
    "td_booking_slot", "td_booking_slot_date", "td_booking_slot_time",
    "td_conducted_date",
    "vehicle_make", "vehicle_model",
    "car_location_at_time_of_sale",
]

STAGE_LABELS = {
    STAGE_ENQUIRY_ID:   "Enquiry (no TD)",
    STAGE_BOOKED_ID:    "TD booked",
    STAGE_CONDUCTED_ID: "TD conducted (no deposit)",
}

# ============ UI Theme ============
st.set_page_config(page_title="Pawan Customer Connector", layout="wide")
PRIMARY = "#4436F5"

st.markdown(f"""
<style>
html, body, [data-testid="stAppViewContainer"] {{
  background: #FFFFFF;
  color: #000000;
}}
.block-container {{ max-width: 1200px !important; }}

.header-title {{ color: {PRIMARY}; }}
hr.div {{ border:0;border-top:1px solid #E5E7EB; margin:12px 0 8px }}

div.stButton > button {{
  background: {PRIMARY} !important;
  color: #FFFFFF !important;
  border: 1px solid {PRIMARY} !important;
  border-radius: 12px;
  font-weight: 600;
}}
div.stButton > button:hover {{ background: {PRIMARY} !important; }}
div.stButton > button.cta {{ width:100%; height:100px; font-size:18px; text-align:left; border-radius:16px; }}

.form-row {{ display:flex; justify-content:center; align-items:end; gap:12px; flex-wrap:wrap; }}

input, select, textarea {{
  background: #FFFFFF !important;
  color: #000000 !important;
  border: 1px solid #D1D5DB !important;
  border-radius: 10px !important;
}}
label, .stSelectbox label, .stDateInput label, .stTextInput label {{ color: #000000 !important; }}

/* Wrap long text inside data editor / dataframes */
[data-testid="stDataFrame"] div[role="cell"] {{
  white-space: pre-wrap !important;
  line-height: 1.4 !important;
}}
[data-testid="stDataFrame"] * {{ color:#000000 !important; }}
[data-testid="stDataFrame"] div[data-testid="stVerticalBlock"] {{ background:#FFFFFF !important; }}
[data-testid="stTable"] td, [data-testid="stTable"] th {{ color:#000000 !important; }}

/* Preview table legacy (kept if we render) */
.preview-table table {{
  background: #FFFFFF !important; color: #000000 !important; border-collapse: collapse !important; width: 100%;
}}
.preview-table th, .preview-table td {{
  border: 1px solid #000000 !important; padding: 8px 12px !important; vertical-align: top !important;
}}
.preview-table th {{ font-weight: 700 !important; }}
</style>
""", unsafe_allow_html=True)

# ============ Helpers ============
def hs_headers() -> dict:
    return {"Authorization": f"Bearer {HUBSPOT_TOKEN}"}

def stage_label(stage_id: str) -> str:
    sid = str(stage_id or "")
    return STAGE_LABELS.get(sid, sid or "")

def mel_day_bounds_to_epoch_ms(d: date) -> tuple[int, int]:
    start_local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=MEL_TZ)
    end_local   = start_local + timedelta(days=1) - timedelta(milliseconds=1)
    start_ms    = int(start_local.astimezone(UTC_TZ).timestamp() * 1000)
    end_ms      = int(end_local.astimezone(UTC_TZ).timestamp() * 1000)
    return start_ms, end_ms

def mel_range_bounds_to_epoch_ms(d1: date, d2: date) -> tuple[int, int]:
    if d2 < d1: d1, d2 = d2, d1
    s,_ = mel_day_bounds_to_epoch_ms(d1)
    _,e = mel_day_bounds_to_epoch_ms(d2)
    return s,e

def parse_epoch_or_iso_to_local_date(s) -> date | None:
    try:
        if s is None or (isinstance(s, float) and np.isnan(s)): return None
        if isinstance(s, (int, np.integer)) or (isinstance(s, str) and s.isdigit()):
            dt = pd.to_datetime(int(s), unit="ms", utc=True).tz_convert(MEL_TZ)
        else:
            dt = pd.to_datetime(s, utc=True)
            if dt.tzinfo is None: dt = dt.tz_localize("UTC")
            dt = dt.tz_convert(MEL_TZ)
        return dt.date()
    except Exception:
        try: return pd.to_datetime(s).date()
        except Exception: return None

def parse_epoch_or_iso_to_local_time(s) -> str:
    try:
        if s is None or (isinstance(s, float) and np.isnan(s)): return ""
        if isinstance(s, (int, np.integer)) or (isinstance(s, str) and s.isdigit()):
            dt = pd.to_datetime(int(s), unit="ms", utc=True).tz_convert(MEL_TZ)
        else:
            dt = pd.to_datetime(s, utc=True)
            if dt.tzinfo is None: dt = dt.tz_localize("UTC")
            dt = dt.tz_convert(MEL_TZ)
        return dt.strftime("%H:%M")
    except Exception:
        return ""

def parse_td_slot_time_prop(val) -> str:
    """Parse HubSpot 'td_booking_slot_time' -> 'HH:MM' local if epoch, or normalize common strings."""
    if val is None or (isinstance(val, float) and np.isnan(val)): return ""
    s = str(val).strip()
    if not s: return ""
    if s.isdigit() and len(s) >= 10:
        try:
            return pd.to_datetime(int(s), unit="ms", utc=True).tz_convert(MEL_TZ).strftime("%H:%M")
        except Exception:
            pass
    for fmt in ["%H:%M", "%I:%M %p", "%H:%M:%S"]:
        try:
            t = datetime.strptime(s, fmt).time()
            return f"{t.hour:02d}:{t.minute:02d}"
        except Exception:
            continue
    try:
        ts = pd.to_datetime(s)
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is None: ts = ts.tz_localize("UTC")
            ts = ts.tz_convert(MEL_TZ)
            return ts.strftime("%H:%M")
    except Exception:
        pass
    return s

def normalize_phone(raw) -> str:
    if pd.isna(raw) or raw is None: return ''
    s = str(raw).strip()
    if s.startswith('+'): digits = '+' + ''.join(ch for ch in s if ch.isdigit())
    else:                 digits = ''.join(ch for ch in s if ch.isdigit())
    if digits.startswith('+61') and len(digits) == 12: return digits
    if digits.startswith('61')  and len(digits) == 11: return '+' + digits
    if digits.startswith('0')   and len(digits) == 10 and digits[1] == '4': return '+61' + digits[1:]
    if digits.startswith('4')   and len(digits) == 9:  return '+61' + digits
    return ''

def format_date_au(d: date) -> str:
    return d.strftime("%d %b %Y") if isinstance(d, date) else ""

def rel_date(d: date) -> str:
    if not isinstance(d, date): return ''
    today = datetime.now(MEL_TZ).date()
    diff = (d - today).days
    if diff == 0: return 'today'
    if diff == 1: return 'tomorrow'
    if diff == -1: return 'yesterday'
    if 1 < diff <= 7: return 'in a few days'
    if -7 <= diff < -1: return 'a few days ago'
    if 8 <= diff <= 14: return 'next week'
    if -14 <= diff <= -8: return 'last week'
    return d.strftime('%b %d')

def first_nonempty_str(series: pd.Series) -> str:
    if series is None: return ""
    s = series.astype(str).fillna("").map(lambda x: x.strip())
    s = s[(s.astype(bool)) & (s.str.lower() != "nan")]
    return s.iloc[0] if not s.empty else ""

def prepare_deals(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame): df = pd.DataFrame()
    else: df = df.copy()
    for c in DEAL_PROPS:
        if c not in df.columns: df[c] = pd.Series(dtype="object")
    df["slot_date"]      = df["td_booking_slot"].apply(parse_epoch_or_iso_to_local_date)
    df["slot_time"]      = df["td_booking_slot"].apply(parse_epoch_or_iso_to_local_time)           # legacy time
    df["slot_date_prop"] = df["td_booking_slot_date"].apply(parse_epoch_or_iso_to_local_date)
    df["slot_time_param"]= df["td_booking_slot_time"].apply(parse_td_slot_time_prop)               # preferred time
    df["conducted_date_local"] = df["td_conducted_date"].apply(parse_epoch_or_iso_to_local_date)
    df["conducted_time_local"] = df["td_conducted_date"].apply(parse_epoch_or_iso_to_local_time)
    df["phone_raw"]      = df["mobile"].where(df["mobile"].notna(), df["phone"])
    df["phone_norm"]     = df["phone_raw"].apply(normalize_phone)
    df["email"]          = df["email"].fillna('')
    df["full_name"]      = df["full_name"].fillna('')
    return df

def filter_internal_test_emails(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove cars24.com / yopmail.com emails. Return (filtered_df, removed_df[with Reason])."""
    if df is None or df.empty or "email" not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(), pd.DataFrame()
    work = df.copy()
    dom = work["email"].astype(str).str.strip().str.lower().str.split("@").str[-1]
    mask = ~dom.isin({"cars24.com", "yopmail.com"})
    removed = work[~mask].copy()
    if not removed.empty:
        removed["Reason"] = "Internal/test email domain"
    return work[mask].copy(), removed

# Add these new functions to your app.py

# HubSpot Cars API endpoints
HS_CARS_SEARCH_URL = f"{HS_ROOT}/crm/v3/objects/cars/search"

def hs_deals_to_cars_map(deal_ids: list[str]) -> dict[str, list[str]]:
    """Get car IDs associated with each deal."""
    out = {str(d): [] for d in deal_ids}
    if not deal_ids: return out
    
    url = f"{HS_ROOT}/crm/v4/objects/deals/batch/read"
    payload = {"properties": [], "inputs": [{"id": str(d)} for d in deal_ids], "associations": ["cars"]}
    try:
        r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
        r.raise_for_status()
        for item in r.json().get("results", []):
            did = str(item.get("id"))
            cars = [a.get("id") for a in item.get("associations", {}).get("cars", [])]
            out[did] = [str(x) for x in cars if x]
    except Exception as e:
        st.warning(f"Could not read deal→cars associations: {e}")
    return out

def hs_cars_to_deals_map(car_ids: list[str]) -> dict[str, list[str]]:
    """Get deal IDs associated with each car."""
    out = {str(c): [] for c in car_ids}
    if not car_ids: return out
    
    url = f"{HS_ROOT}/crm/v4/objects/cars/batch/read"
    payload = {"properties": [], "inputs": [{"id": str(c)} for c in car_ids], "associations": ["deals"]}
    try:
        r = requests.post(url, headers=hs_headers(), json=payload, timeout=25)
        r.raise_for_status()
        for item in r.json().get("results", []):
            cid = str(item.get("id"))
            deals = [a.get("id") for a in item.get("associations", {}).get("deals", [])]
            out[cid] = [str(x) for x in deals if x]
    except Exception as e:
        st.warning(f"Could not read car→deals associations: {e}")
    return out

def filter_deals_by_car_active_purchases(deals_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out deals where the associated car has other deals with active purchase stages.
    Returns (kept_deals, dropped_deals_with_reason).
    """
    if deals_df is None or deals_df.empty:
        return deals_df.copy() if isinstance(deals_df, pd.DataFrame) else pd.DataFrame(), pd.DataFrame()
    
    st.write(f"🔍 DEBUG: Starting car filter with {len(deals_df)} deals")
    
    # Get deal IDs
    deal_ids = deals_df.get("hs_object_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
    if not deal_ids:
        st.write("🔍 DEBUG: No deal IDs found")
        return deals_df.copy(), pd.DataFrame()
    
    st.write(f"🔍 DEBUG: Processing deal IDs: {deal_ids}")
    
    # Get deal → car associations
    d2c = hs_deals_to_cars_map(deal_ids)
    st.write(f"🔍 DEBUG: Deal→Car mapping: {d2c}")
    
    # Get all unique car IDs
    car_ids = sorted({cid for cids in d2c.values() for cid in cids})
    if not car_ids:
        st.write("🔍 DEBUG: No car IDs found in associations")
        return deals_df.copy(), pd.DataFrame()
    
    st.write(f"🔍 DEBUG: Found car IDs: {car_ids}")
    
    # Get car → deals associations
    c2d = hs_cars_to_deals_map(car_ids)
    st.write(f"🔍 DEBUG: Car→Deal mapping: {c2d}")
    
    # Get all other deal IDs (not in our original set)
    other_deal_ids = sorted({did for _, dlist in c2d.items() for did in dlist if did not in deal_ids})
    st.write(f"🔍 DEBUG: Other deal IDs to check: {other_deal_ids}")
    
    if not other_deal_ids:
        st.write("🔍 DEBUG: No other deals found to check")
        return deals_df.copy(), pd.DataFrame()
    
    # Get stages for other deals
    stage_map = hs_batch_read_deals(other_deal_ids, props=["dealstage"])
    st.write(f"🔍 DEBUG: Stage mapping for other deals: {stage_map}")
    
    # Find cars that have active purchase deals
    cars_with_active_purchases = set()
    for cid, dlist in c2d.items():
        for did in dlist:
            if did in deal_ids:  # Skip our original deals
                continue
            stage = (stage_map.get(did, {}) or {}).get("dealstage")
            st.write(f"🔍 DEBUG: Deal {did} has stage {stage}")
            if stage and str(stage) in ACTIVE_PURCHASE_STAGE_IDS:
                cars_with_active_purchases.add(cid)
                st.write(f"🔍 DEBUG: Car {cid} has active purchase (deal {did} with stage {stage})")
    
    st.write(f"🔍 DEBUG: Cars with active purchases: {cars_with_active_purchases}")
    
    # Filter deals
    def keep_deal(row):
        d_id = str(row.get("hs_object_id") or "")
        cids = d2c.get(d_id, [])
        should_keep = not any((c in cars_with_active_purchases) for c in cids)
        st.write(f"🔍 DEBUG: Deal {d_id} with cars {cids} -> keep: {should_keep}")
        return should_keep
    
    work = deals_df.copy()
    work["__keep"] = work.apply(keep_deal, axis=1)
    
    dropped = work[~work["__keep"]].drop(columns=["__keep"]).copy()
    kept = work[work["__keep"]].drop(columns=["__keep"]).copy()
    
    if not dropped.empty:
        dropped["Reason"] = "Car has another deal in active purchase stage"
    
    st.write(f"🔍 DEBUG: Final result - Kept: {len(kept)}, Dropped: {len(dropped)}")
    
    return kept, dropped


def show_removed_table(df: pd.DataFrame, title: str):
    """Small helper to render removed items table (if any)."""
    if df is None or df.empty:
        return
    cols = [c for c in ["full_name","email","phone_norm","vehicle_make","vehicle_model","dealstage","hs_object_id","Reason"]
            if c in df.columns]
    st.markdown(f"**{title}** ({len(df)})")
    st.dataframe(df[cols]
                 .rename(columns={
                     "full_name":"Customer","phone_norm":"Phone",
                     "vehicle_make":"Make","vehicle_model":"Model",
                     "dealstage":"Stage"
                 }),
                 use_container_width=True)


def dedupe_users_with_audit(df: pd.DataFrame, *, use_conducted: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Same as dedupe_users, but also returns a DataFrame of deals 'removed' by dedupe
    (i.e., additional deals beyond the first per user_key), with a Reason.
    """
    base = dedupe_users(df, use_conducted=use_conducted)
    if df is None or df.empty:
        return base, pd.DataFrame()
    work = df.copy()
    work["email_l"] = work["email"].astype(str).str.strip().str.lower()
    work["user_key"] = (work["phone_norm"].fillna('') + "|" + work["email_l"].fillna('')).str.strip()
    work = work[work["user_key"].astype(bool)]
    dropped_rows = []
    for _, grp in work.groupby("user_key", sort=False):
        if len(grp) <= 1:
            continue
        # We consider the first row as the representative; the rest are 'collapsed' by dedupe
        representative = grp.iloc[0]
        rep_name  = str(representative.get("full_name") or "").strip()
        rep_phone = str(representative.get("phone_norm") or "").strip()
        rep_email = str(representative.get("email") or "").strip()
        for _, r in grp.iloc[1:].iterrows():
            dropped_rows.append({
                "hs_object_id": r.get("hs_object_id"),
                "full_name": r.get("full_name"),
                "email": r.get("email"),
                "phone_norm": r.get("phone_norm"),
                "vehicle_make": r.get("vehicle_make"),
                "vehicle_model": r.get("vehicle_model"),
                "dealstage": r.get("dealstage"),
                "Reason": f"Deduped under {rep_name or rep_phone or rep_email}"
            })
    dropped_df = pd.DataFrame(dropped_rows)
    return base, dropped_df


def build_messages_with_audit(dedup_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build messages; also return a 'skipped' DF with reasons (e.g., missing phone, empty draft).
    """
    msgs_df = build_messages_from_dedup(dedup_df, mode=mode)
    skipped = []
    # Re-run minimal checks so we can collect reasons for any that were dropped
    if dedup_df is not None and not dedup_df.empty:
        for _, row in dedup_df.iterrows():
            phone = str(row.get("Phone") or "").strip()
            if not phone:
                skipped.append({
                    "Customer": str(row.get("CustomerName") or ""),
                    "Email": str(row.get("Email") or ""),
                    "Cars": str(row.get("Cars") or ""),
                    "Reason": "Missing/invalid phone"
                })
            else:
                # Find the corresponding message row; if none, we skipped it
                if msgs_df.empty or not (msgs_df["Phone"] == phone).any():
                    skipped.append({
                        "Customer": str(row.get("CustomerName") or ""),
                        "Email": str(row.get("Email") or ""),
                        "Cars": str(row.get("Cars") or ""),
                        "Reason": "No message generated"
                    })
    skipped_df = pd.DataFrame(skipped, columns=["Customer","Email","Cars","Reason"])
    return msgs_df, skipped_df

# ============ HubSpot ============
@st.cache_data(show_spinner=False)
def hs_get_deal_property_options(property_name: str) -> list[dict]:
    # Fallback to keep app usable if network flakes
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

def _search_once(payload: dict, total_cap: int) -> pd.DataFrame:
    results, fetched, after = [], 0, None
    while True:
        try:
            if after: payload["after"] = after
            r = requests.post(HS_SEARCH_URL, headers=hs_headers(), json=payload, timeout=25)
            if r.status_code != 200:
                try: msg = r.json()
                except Exception: msg = {"error": r.text}
                st.error(f"HubSpot search error {r.status_code}: {msg}")
                break
            data = r.json()
            for item in data.get("results", []):
                results.append(item.get("properties", {}) or {})
                fetched += 1
                if fetched >= total_cap: break
            if fetched >= total_cap: break
            after = (data.get("paging") or {}).get("next", {}).get("after")
            if not after: break
            time.sleep(0.08)
        except Exception as e:
            st.error(f"Network/search error: {e}")
            break
    return pd.DataFrame(results) if results else pd.DataFrame(columns=DEAL_PROPS)

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

def hs_search_deals_by_appointment_and_stages(appointment_id: str, pipeline_id: str, stage_ids: set[str]) -> pd.DataFrame:
    filters = [
        {"propertyName": "pipeline", "operator": "EQ", "value": pipeline_id},
        {"propertyName": "appointment_id", "operator": "EQ", "value": str(appointment_id).strip()},
        {"propertyName": "dealstage", "operator": "IN", "values": list(stage_ids)},
    ]
    payload = {"filterGroups": [{"filters": filters}], "properties": DEAL_PROPS, "limit": HS_PAGE_LIMIT}
    return _search_once(payload, total_cap=HS_TOTAL_CAP)

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
def send_sms_via_aircall(phone: str, message: str) -> tuple[bool, str]:
    try:
        url = f"{AIRCALL_BASE_URL}/numbers/{AIRCALL_NUMBER_ID}/messages/native/send"
        resp = requests.post(url, json={"to": phone, "body": message}, auth=(AIRCALL_ID, AIRCALL_TOKEN), timeout=12)
        resp.raise_for_status()
        return True, "sent"
    except Exception as e:
        return False, str(e)

# ============ OpenAI drafting ============
def _call_openai(messages):
    if not _openai_ok or not OPENAI_API_KEY or openai is None:
        return ""
    try:
        if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
            for model in PREFERRED_MODELS:
                try:
                    resp = openai.chat.completions.create(model=model, messages=messages, temperature=0.6, max_tokens=180)
                    return resp.choices[0].message.content.strip()
                except Exception:
                    continue
    except Exception:
        pass
    try:
        if hasattr(openai, "ChatCompletion"):
            for model in PREFERRED_MODELS:
                try:
                    resp = openai.ChatCompletion.create(model=model, messages=messages, temperature=0.6, max_tokens=180)
                    return resp["choices"][0]["message"]["content"].strip()
                except Exception:
                    continue
    except Exception:
        pass
    return ""

def draft_sms_reminder(name: str, pairs_text: str) -> str:
    system = (
        "You write outbound SMS for Cars24 Laverton (Australia). "
        "Tone: warm, polite, inviting, Australian. AU spelling. "
        "Keep ~280 chars. No emojis/links. Avoid apostrophes. "
        "Write as the business (sender). Include a clear CTA to confirm or reschedule."
    )
    user = f"Recipient name: {name or 'there'}.\nUpcoming test drive(s): {pairs_text}.\nFriendly reminder."
    text = _call_openai([
        {"role":"system","content":system},
        {"role":"user","content":user}
    ]) or ""
    return text if text.endswith("–Cars24 Laverton") else f"{text} –Cars24 Laverton".strip()

def draft_sms_manager(name: str, pairs_text: str) -> str:
    first = (name or "").split()[0] if (name or "").strip() else "there"
    system = (
        "You write outbound SMS for Cars24 Laverton (Australia) from the store manager, Pawan. "
        "Context: the customer completed a test drive. "
        "Tone: warm, courteous, Australian; encourage a reply. "
        "Goal: ask if they want to proceed (deposit/next steps), offer help, invite brief feedback. "
        "Keep ~300 chars. No emojis/links. Avoid apostrophes."
    )
    user = (
        f"Recipient name: {name or 'there'}.\n"
        f"Completed test drive(s): {pairs_text}.\n"
        f"Begin the SMS with exactly: Hi {first}, this is Pawan, Sales Manager at Cars24 Laverton.\n"
        "Then ask about proceeding (deposit/next steps), offer assistance, invite quick feedback."
    )
    text = _call_openai([
        {"role":"system","content":system},
        {"role":"user","content":user}
    ]) or ""
    intro = f"hi {first.lower()}, this is pawan, sales manager at cars24 laverton"
    if text.strip().lower().startswith(intro): return text.strip()
    return f"{text.strip()} –Pawan, Sales Manager"

def draft_sms_oldlead_by_stage(name: str, car_text: str, stage_hint: str) -> str:
    first = (name or "").split()[0] if (name or "").strip() else "there"
    if stage_hint == "enquiry":
        context = "They enquired but have not booked a test drive."
        ask = "Invite them to book a test drive at a time that suits and offer personal help."
    elif stage_hint == "booked":
        context = "They booked a test drive but it did not go ahead."
        ask = "Invite them to reschedule the drive and offer personal help."
    elif stage_hint == "conducted":
        context = "They completed a test drive but did not proceed."
        ask = "Ask if they would like to move forward (deposit/next steps) and offer assistance."
    else:
        context = "It has been a while since they reached out."
        ask = "Invite them back, offer personal help, and check interest in moving forward."
    system = (
        "You write outbound SMS for Cars24 Laverton (Australia) from the store manager, Pawan. "
        "Tone: warm, courteous, Australian; avoid pressure; encourage a reply. "
        "Promise personal attention and that we will work out a deal they will love. "
        "Keep ~300 characters. No emojis/links. Avoid apostrophes."
    )
    user = (
        f"Recipient name: {name or 'there'}.\n"
        f"Car(s) of interest: {car_text}.\n"
        f"Stage context: {context}\n"
        f"Begin the SMS with exactly: Hi {first}, this is Pawan, Sales Manager at Cars24 Laverton.\n"
        f"{ask} Make it friendly and concise."
    )
    text = _call_openai([
        {"role":"system","content":system},
        {"role":"user","content":user}
    ]) or ""
    intro = f"hi {first.lower()}, this is pawan, sales manager at cars24 laverton"
    if text.strip().lower().startswith(intro): return text.strip()
    return f"{text.strip()} –Pawan, Sales Manager"

# ============ Dedupe & SMS build ============
def dedupe_users(df: pd.DataFrame, *, use_conducted: bool) -> pd.DataFrame:
    """Return rows with: CustomerName, Phone, Email, DealsCount, Cars, WhenExact, WhenRel, DealStages, StageHint."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","WhenRel","DealStages","StageHint"])
    work = df.copy()
    work["email_l"] = work["email"].astype(str).str.strip().str.lower()
    work["user_key"] = (work["phone_norm"].fillna('') + "|" + work["email_l"].fillna('')).str.strip()
    work = work[work["user_key"].astype(bool)]
    rows = []
    for _, grp in work.groupby("user_key", sort=False):
        name  = first_nonempty_str(grp["full_name"])
        phone = first_nonempty_str(grp["phone_norm"])
        email = first_nonempty_str(grp["email"])
        cars_list, when_exact_list, when_rel_list, stages_list = [], [], [], []
        for _, r in grp.iterrows():
            car = f"{str(r.get('vehicle_make') or '').strip()} {str(r.get('vehicle_model') or '').strip()}".strip() or "car"
            if use_conducted:
                d = r.get("conducted_date_local"); t = r.get("conducted_time_local") or ""
            else:
                d = r.get("slot_date_prop") or r.get("slot_date")
                t = r.get("slot_time_param") or r.get("slot_time") or ""
            when_rel = rel_date(d) if isinstance(d, date) else ""
            when_exact = (f"{format_date_au(d)} {t}".strip()).strip()
            cars_list.append(car)
            when_exact_list.append(when_exact)
            when_rel_list.append(when_rel if t == "" else f"{when_rel} at {t}".strip())
            stages_list.append(str(r.get("dealstage") or ""))
        stage_labels = sorted({stage_label(x) for x in stages_list if str(x)})
        if STAGE_CONDUCTED_ID in stages_list: hint = "conducted"
        elif STAGE_BOOKED_ID in stages_list: hint = "booked"
        elif STAGE_ENQUIRY_ID in stages_list: hint = "enquiry"
        else: hint = "unknown"
        rows.append({
            "CustomerName": name, "Phone": phone, "Email": email,
            "DealsCount": len([c for c in cars_list if c]),
            "Cars": "; ".join([c for c in cars_list if c]),
            "WhenExact": "; ".join([w for w in when_exact_list if w]),
            "WhenRel": "; ".join([w for w in when_rel_list if w]),
            "DealStages": "; ".join(stage_labels) if stage_labels else "",
            "StageHint": hint
        })
    out = pd.DataFrame(rows)
    want = ["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","WhenRel","DealStages","StageHint"]
    return out[want] if not out.empty else out

def build_pairs_text(cars: str, when_rel: str) -> str:
    c_list = [c.strip() for c in (cars or "").split(";") if c.strip()]
    w_list = [w.strip() for w in (when_rel or "").split(";") if w.strip()]
    pairs = []
    for i in range(max(len(c_list), len(w_list))):
        c = c_list[i] if i < len(c_list) else ""
        w = w_list[i] if i < len(w_list) else ""
        pairs.append(f"{c} {w}".strip())
    return "; ".join([p for p in pairs if p])

def build_messages_from_dedup(dedup_df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if dedup_df is None or dedup_df.empty:
        return pd.DataFrame(columns=["CustomerName","Phone","Email","Cars","WhenExact","WhenRel","DealStages","Message"])
    out = []
    for _, row in dedup_df.iterrows():
        phone = str(row.get("Phone") or "").strip()
        if not phone: continue
        name  = str(row.get("CustomerName") or "").strip()
        cars  = str(row.get("Cars") or "").strip()
        when_rel = str(row.get("WhenRel") or "").strip()
        pairs_text = build_pairs_text(cars, when_rel)
        if mode == "reminder":
            msg = draft_sms_reminder(name, pairs_text)
        elif mode == "manager":
            msg = draft_sms_manager(name, pairs_text)
        else:
            car_text = cars or "the car you were eyeing"
            stage_hint = str(row.get("StageHint") or "unknown")
            msg = draft_sms_oldlead_by_stage(name, car_text, stage_hint)
        out.append({"CustomerName": name, "Phone": phone, "Email": str(row.get("Email") or "").strip(),
                    "Cars": cars, "WhenExact": str(row.get("WhenExact") or ""), "WhenRel": when_rel,
                    "DealStages": str(row.get("DealStages") or ""), "Message": msg})
    return pd.DataFrame(out, columns=["CustomerName","Phone","Email","Cars","WhenExact","WhenRel","DealStages","Message"])

# ============ Rendering helpers ============
def header():
    cols = st.columns([1, 6, 1.2])
    with cols[0]:
        logo_path = next((p for p in ["H2.svg", "cars24_logo.svg", "logo.svg", "cars24.png"] if os.path.exists(p)), None)
        if logo_path: st.image(logo_path, width=100, use_container_width=False)
        else:
            st.markdown(
                f"<div style='height:40px;display:flex;align-items:center;'><div style='background:{PRIMARY};padding:6px 10px;border-radius:6px;'><span style='font-weight:800;color:#FFFFFF'>CARS24</span></div></div>",
                unsafe_allow_html=True
            )
    with cols[1]:
        st.markdown('<h1 class="header-title" style="margin:0;">Pawan Customer Connector</h1>', unsafe_allow_html=True)
    with cols[2]:
        if st.session_state.get("view","home")!="home":
            if st.button("← Back", key="back_btn", use_container_width=True):
                st.session_state["view"]="home"
    st.markdown('<hr class="div"/>', unsafe_allow_html=True)

def ctas():
    c1,c2,c3 = st.columns(3)
    with c1:
        if st.button("🛣️  Test Drive Reminders", key="cta1"):
            st.session_state["view"]="reminders"
    with c2:
        if st.button("👔  Manager Follow-Ups\n\n• After TD conducted  • Single date or range", key="cta2"):
            st.session_state["view"]="manager"
    with c3:
        if st.button("🕰️  Old Leads by Appointment ID\n\n• Re-engage older enquiries  • Skips active purchases", key="cta3"):
            st.session_state["view"]="old"
    st.markdown("""
    <script>
      const btns = window.parent.document.querySelectorAll('button[kind="secondary"]');
      btns.forEach(b => { b.classList.add('cta'); });
    </script>
    """, unsafe_allow_html=True)

def render_trimmed(df: pd.DataFrame, title: str, cols_map: list[tuple[str,str]]):
    st.markdown(f"#### <span style='color:#000000;'>{title}</span>", unsafe_allow_html=True)
    if df is None or df.empty:
        st.info("No rows to show."); return
    disp = df.copy()
    if "dealstage" in disp.columns and "Stage" not in disp.columns:
        disp["Stage"] = disp["dealstage"].apply(stage_label)
    selected, rename = [], {}
    for col,label in cols_map:
        if col in disp.columns:
            selected.append(col)
            if label and label != col: rename[col] = label
        elif col == "Stage" and "Stage" in disp.columns:
            selected.append("Stage")
            rename["Stage"] = label or "Stage"
    st.dataframe(disp[selected].rename(columns=rename), use_container_width=True)

def render_selectable_messages(messages_df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Shows a data_editor with a checkbox per row; returns the edited DF.
       - Default = UNCHECKED
       - Forces text wrapping in the 'SMS draft' column for readability
    """
    if messages_df is None or messages_df.empty:
        st.info("No messages to preview."); return pd.DataFrame()

    # Enhanced CSS for better text wrapping
    st.markdown("""
    <style>
    /* Force text wrapping in all data editor cells */
    [data-testid="stDataEditor"] div[role="gridcell"] {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        word-break: break-word !important;
        overflow-wrap: anywhere !important;
        line-height: 1.4 !important;
        max-width: none !important;
        height: auto !important;
        min-height: 60px !important;
    }
    
    /* Specific targeting for SMS draft column */
    [data-testid="stDataEditor"] div[role="gridcell"]:nth-child(4) {
        white-space: pre-wrap !important;
        word-wrap: break-word !important;
        overflow: visible !important;
        text-overflow: unset !important;
        min-height: 80px !important;
        padding: 8px !important;
    }
    
    /* Allow rows to expand */
    [data-testid="stDataEditor"] div[role="row"] {
        align-items: stretch !important;
        height: auto !important;
        min-height: 60px !important;
    }
    
    /* Grid container adjustments */
    [data-testid="stDataEditor"] div[role="grid"] {
        overflow: visible !important;
    }
    
    /* Column header adjustments */
    [data-testid="stDataEditor"] div[role="columnheader"] {
        height: auto !important;
        min-height: 40px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    view_df = messages_df[["CustomerName","Phone","Message"]].rename(
        columns={"CustomerName":"Customer","Message":"SMS draft"}
    ).copy()
    if "Send" not in view_df.columns:
        view_df.insert(0, "Send", False)  # default UNCHECKED

    edited = st.data_editor(
        view_df,
        key=f"editor_{key}",
        use_container_width=True,
        height=400,  # Give more height for the editor
        column_config={
            "Send": st.column_config.CheckboxColumn("Send", help="Tick to send this SMS", default=False, width="small"),
            "Customer": st.column_config.TextColumn("Customer", width=150),
            "Phone": st.column_config.TextColumn("Phone", width=130),
            "SMS draft": st.column_config.TextColumn("SMS draft", width=500, help="Click to edit message"),
        },
        hide_index=True,
    )
    return edited


# ============ Views (persist data in session_state) ============
def view_reminders():
    st.subheader("🛣️  Test Drive Reminders")
    with st.form("reminders_form"):
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        c1,c2,c3 = st.columns([2,2,1])
        with c1: rem_date = st.date_input("TD booking date", value=datetime.now(MEL_TZ).date())
        state_options = hs_get_deal_property_options("car_location_at_time_of_sale")
        values = [o["value"] for o in state_options] if state_options else []
        labels = [o["label"] for o in state_options] if state_options else []
        def_val = "VIC" if "VIC" in values else (values[0] if values else "")
        with c2:
            if labels:
                chosen_label = st.selectbox("Vehicle state", labels, index=(values.index("VIC") if "VIC" in values else 0))
                label_to_val = {o["label"]:o["value"] for o in state_options}
                rem_state_val = label_to_val.get(chosen_label, def_val)
            else:
                rem_state_val = st.text_input("Vehicle state", value=def_val)
        with c3: go = st.form_submit_button("Fetch deals", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if go:
        st.markdown("<span style='background:#4436F5;color:#FFFFFF;padding:4px 8px;border-radius:6px;'>Searching HubSpot…</span>", unsafe_allow_html=True)
        eq_ms, _ = mel_day_bounds_to_epoch_ms(rem_date)
        raw = hs_search_deals_by_date_property(
            pipeline_id=PIPELINE_ID, stage_id=STAGE_BOOKED_ID, state_value=rem_state_val,
            date_property="td_booking_slot_date", date_eq_ms=eq_ms,
            date_start_ms=None, date_end_ms=None, total_cap=HS_TOTAL_CAP
        )
        deals = prepare_deals(raw)

        # 1) NEW: Filter by car active purchases (before other filters) - WITH DEBUGGING
        deals_car_filtered, dropped_car_purchases = filter_deals_by_car_active_purchases(deals)

        # 2) Filter internal/test emails + show callout
        deals_f, removed_internal = filter_internal_test_emails(deals_car_filtered)

        # 3) Audit dedupe (returns dedup + dropped list with reasons)
        dedup, dedupe_dropped = dedupe_users_with_audit(deals_f, use_conducted=False)

        # 4) Build messages with audit (messages + skipped list with reasons)
        msgs, skipped_msgs = build_messages_with_audit(dedup, mode="reminder")

        # persist all artifacts so checkboxes don't reset the page
        st.session_state["reminders_deals"] = deals_f
        st.session_state["reminders_dropped_car_purchases"] = dropped_car_purchases
        st.session_state["reminders_removed_internal"] = removed_internal
        st.session_state["reminders_dedup"] = dedup
        st.session_state["reminders_dedupe_dropped"] = dedupe_dropped
        st.session_state["reminders_msgs"]  = msgs
        st.session_state["reminders_skipped_msgs"] = skipped_msgs

    # ---- Render from persisted state ----
    deals_f      = st.session_state.get("reminders_deals")
    dropped_car  = st.session_state.get("reminders_dropped_car_purchases")
    removed_int  = st.session_state.get("reminders_removed_internal")
    dedup        = st.session_state.get("reminders_dedup")
    dedupe_drop  = st.session_state.get("reminders_dedupe_dropped")
    msgs         = st.session_state.get("reminders_msgs")
    skipped_msgs = st.session_state.get("reminders_skipped_msgs")

    # Show car purchase filter results FIRST
    if isinstance(dropped_car, pd.DataFrame) and not dropped_car.empty:
        show_removed_table(dropped_car, "Removed (car has active purchase deal)")

    if isinstance(removed_int, pd.DataFrame) and not removed_int.empty:
        show_removed_table(removed_int, "Removed by domain filter (cars24.com / yopmail.com)")

    if isinstance(deals_f, pd.DataFrame) and not deals_f.empty:
        render_trimmed(deals_f, "Filtered deals (trimmed)", [
            ("hs_object_id","Deal ID"), ("full_name","Customer"), ("email","Email"), ("phone_norm","Phone"),
            ("vehicle_make","Make"), ("vehicle_model","Model"),
            ("slot_date_prop","TD booking date"), ("slot_time_param","Time"),
            ("Stage","Stage"),
        ])

    if isinstance(dedupe_drop, pd.DataFrame) and not dedupe_drop.empty:
        show_removed_table(dedupe_drop, "Collapsed during dedupe (duplicates)")

    if isinstance(dedup, pd.DataFrame) and not dedup.empty:
        st.markdown("#### <span style='color:#000000;'>Deduped list (by mobile|email)</span>", unsafe_allow_html=True)
        st.dataframe(dedup[["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","DealStages"]]
                     .rename(columns={"WhenExact":"When (exact)","DealStages":"Stage(s)"}),
                     use_container_width=True)

    if isinstance(msgs, pd.DataFrame) and not msgs.empty:
        st.markdown("#### <span style='color:#000000;'>Message Preview (Reminders)</span>", unsafe_allow_html=True)
        edited = render_selectable_messages(msgs, key="reminders")
        if isinstance(skipped_msgs, pd.DataFrame) and not skipped_msgs.empty:
            st.markdown("**Skipped while creating SMS**")
            st.dataframe(skipped_msgs, use_container_width=True)

        if not edited.empty and st.button("Send SMS"):
            to_send = edited[edited["Send"]]
            if to_send.empty:
                st.warning("No rows selected.")
            elif not (AIRCALL_ID and AIRCALL_TOKEN and AIRCALL_NUMBER_ID):
                st.error("Missing Aircall credentials in .env.")
            else:
                st.info("Sending messages…")
                sent, failed = 0, 0
                for _, r in to_send.iterrows():
                    ok, msg = send_sms_via_aircall(r["Phone"], r["SMS draft"])
                    if ok: sent += 1; st.success(f"✅ Sent to {r['Phone']}")
                    else:  failed += 1; st.error(f"❌ Failed for {r['Phone']}: {msg}")
                    time.sleep(1)
                if sent: st.balloons()
                st.success(f"🎉 Done! Sent: {sent} | Failed: {failed}")

def view_manager():
    st.subheader("👔  Manager Follow-Ups")
    with st.form("manager_form"):
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns([1.4,1.6,1.6,1.2])
        with c1: mode = st.radio("Mode", ["Single date","Date range"], horizontal=True, index=1)
        today = datetime.now(MEL_TZ).date()
        if mode=="Single date":
            with c2: d1 = st.date_input("Date", value=today); d2 = d1
            with c3: pass
        else:
            with c2: d1 = st.date_input("Start date", value=today - timedelta(days=7))
            with c3: d2 = st.date_input("End date",   value=today)
        state_options = hs_get_deal_property_options("car_location_at_time_of_sale")
        values = [o["value"] for o in state_options] if state_options else []
        labels = [o["label"] for o in state_options] if state_options else []
        def_val = "VIC" if "VIC" in values else (values[0] if values else "")
        with c4:
            if labels:
                chosen_label = st.selectbox("Vehicle state", labels, index=(values.index("VIC") if "VIC" in values else 0))
                label_to_val = {o["label"]:o["value"] for o in state_options}
                mgr_state_val = label_to_val.get(chosen_label, def_val)
            else:
                mgr_state_val = st.text_input("Vehicle state", value=def_val)
        go = st.form_submit_button("Fetch deals", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if go:
        st.markdown("<span style='background:#4436F5;color:#FFFFFF;padding:4px 8px;border-radius:6px;'>Searching HubSpot…</span>", unsafe_allow_html=True)

        # 1) Base search
        s_ms, e_ms = mel_range_bounds_to_epoch_ms(d1, d2)
        raw = hs_search_deals_by_date_property(
            pipeline_id=PIPELINE_ID, stage_id=STAGE_CONDUCTED_ID, state_value=mgr_state_val,
            date_property="td_conducted_date", date_eq_ms=None,
            date_start_ms=s_ms, date_end_ms=e_ms, total_cap=HS_TOTAL_CAP
        )
        deals0 = prepare_deals(raw)

        # 2) Exclude contacts with other ACTIVE purchase deals
        kept = deals0.copy()
        if not kept.empty:
            deal_ids = kept.get("hs_object_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
            d2c = hs_deals_to_contacts_map(deal_ids)
            contact_ids = sorted({cid for cids in d2c.values() for cid in cids})
            c2d = hs_contacts_to_deals_map(contact_ids)
            other_deal_ids = sorted({did for _, dlist in c2d.items() for did in dlist if did not in deal_ids})
            stage_map = hs_batch_read_deals(other_deal_ids, props=["dealstage"])

            exclude_contacts = set()
            for cid, dlist in c2d.items():
                for did in dlist:
                    if did in deal_ids: continue
                    stage = (stage_map.get(did, {}) or {}).get("dealstage")
                    if stage and str(stage) in ACTIVE_PURCHASE_STAGE_IDS:
                        exclude_contacts.add(cid); break

            def keep_row(row):
                d_id = str(row.get("hs_object_id") or "")
                cids = d2c.get(d_id, [])
                return not any((c in exclude_contacts) for c in cids)

            kept["__keep"] = kept.apply(keep_row, axis=1)
            dropped_active = kept[~kept["__keep"]].drop(columns=["__keep"]).copy()
            kept = kept[kept["__keep"]].drop(columns=["__keep"]).copy()
            if not dropped_active.empty:
                dropped_active["Reason"] = "Contact has another active purchase deal"
                show_removed_table(dropped_active, "Removed (active purchase on another deal)")

        # 3) Filter internal/test emails + callout
        deals_f, removed_internal = filter_internal_test_emails(kept)

        # 4) Audit dedupe
        dedup, dedupe_dropped = dedupe_users_with_audit(deals_f, use_conducted=True)

        # 5) Build messages + audit
        msgs, skipped_msgs = build_messages_with_audit(dedup, mode="manager")

        # persist
        st.session_state["manager_deals"] = deals_f
        st.session_state["manager_removed_internal"] = removed_internal
        st.session_state["manager_dedup"] = dedup
        st.session_state["manager_dedupe_dropped"] = dedupe_dropped
        st.session_state["manager_msgs"]  = msgs
        st.session_state["manager_skipped_msgs"] = skipped_msgs

    deals_f      = st.session_state.get("manager_deals")
    removed_int  = st.session_state.get("manager_removed_internal")
    dedup        = st.session_state.get("manager_dedup")
    dedupe_drop  = st.session_state.get("manager_dedupe_dropped")
    msgs         = st.session_state.get("manager_msgs")
    skipped_msgs = st.session_state.get("manager_skipped_msgs")

    if isinstance(removed_int, pd.DataFrame) and not removed_int.empty:
        show_removed_table(removed_int, "Removed by domain filter (cars24.com / yopmail.com)")

    if isinstance(deals_f, pd.DataFrame) and not deals_f.empty:
        render_trimmed(deals_f, "Filtered deals (trimmed)", [
            ("full_name","Customer"), ("email","Email"),
            ("vehicle_make","Make"), ("vehicle_model","Model"),
            ("conducted_date_local","TD conducted (date)"), ("conducted_time_local","Time"),
            ("Stage","Stage"),
        ])

    if isinstance(dedupe_drop, pd.DataFrame) and not dedupe_drop.empty:
        show_removed_table(dedupe_drop, "Collapsed during dedupe (duplicates)")

    if isinstance(dedup, pd.DataFrame) and not dedup.empty:
        st.markdown("#### <span style='color:#000000;'>Deduped list (by mobile|email)</span>", unsafe_allow_html=True)
        st.dataframe(dedup[["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","DealStages"]]
                     .rename(columns={"WhenExact":"When (exact)","DealStages":"Stage(s)"}),
                     use_container_width=True)

    if isinstance(msgs, pd.DataFrame) and not msgs.empty:
        st.markdown("#### <span style='color:#000000;'>Message Preview (Manager Follow-Ups)</span>", unsafe_allow_html=True)
        edited = render_selectable_messages(msgs, key="manager")
        if isinstance(skipped_msgs, pd.DataFrame) and not skipped_msgs.empty:
            st.markdown("**Skipped while creating SMS**")
            st.dataframe(skipped_msgs, use_container_width=True)

        if not edited.empty and st.button("Send SMS"):
            to_send = edited[edited["Send"]]
            if to_send.empty:
                st.warning("No rows selected.")
            elif not (AIRCALL_ID and AIRCALL_TOKEN and AIRCALL_NUMBER_ID):
                st.error("Missing Aircall credentials in .env.")
            else:
                st.info("Sending messages…")
                sent, failed = 0, 0
                for _, r in to_send.iterrows():
                    ok, msg = send_sms_via_aircall(r["Phone"], r["SMS draft"])
                    if ok: sent += 1; st.success(f"✅ Sent to {r['Phone']}")
                    else:  failed += 1; st.error(f"❌ Failed for {r['Phone']}: {msg}")
                    time.sleep(1)
                if sent: st.balloons()
                st.success(f"🎉 Done! Sent: {sent} | Failed: {failed}")

def view_old():
    st.subheader("🕰️  Old Leads by Appointment ID")
    with st.form("old_form"):
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        c1,c2 = st.columns([2,1])
        with c1: appt = st.text_input("Appointment ID", value="", placeholder="APPT-12345")
        with c2: go = st.form_submit_button("Fetch old leads", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if go:
        if not appt.strip():
            st.error("Please enter an Appointment ID.")
        else:
            st.markdown("<span style='background:#4436F5;color:#FFFFFF;padding:4px 8px;border-radius:6px;'>Searching HubSpot…</span>", unsafe_allow_html=True)
            deals_raw = hs_search_deals_by_appointment_and_stages(
                appointment_id=appt, pipeline_id=PIPELINE_ID, stage_ids=OLD_LEAD_START_STAGES
            )
            deals = prepare_deals(deals_raw)

            # Exclude contacts with other ACTIVE purchase deals (existing logic)
            deal_ids = deals.get("hs_object_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
            d2c = hs_deals_to_contacts_map(deal_ids)
            contact_ids = sorted({cid for cids in d2c.values() for cid in cids})
            c2d = hs_contacts_to_deals_map(contact_ids)
            other_deal_ids = sorted({did for _, dlist in c2d.items() for did in dlist if did not in deal_ids})
            stage_map = hs_batch_read_deals(other_deal_ids, props=["dealstage"])

            exclude_contacts = set()
            for cid, dlist in c2d.items():
                for did in dlist:
                    if did in deal_ids: continue
                    stage = (stage_map.get(did, {}) or {}).get("dealstage")
                    if stage and str(stage) in ACTIVE_PURCHASE_STAGE_IDS:
                        exclude_contacts.add(cid); break

            def keep_row(row):
                d_id = str(row.get("hs_object_id") or "")
                cids = d2c.get(d_id, [])
                return not any((c in exclude_contacts) for c in cids)

            kept = deals.copy()
            if not deals.empty:
                kept["__keep"] = kept.apply(keep_row, axis=1)
                dropped_active = kept[~kept["__keep"]].drop(columns=["__keep"]).copy()
                kept = kept[kept["__keep"]].drop(columns=["__keep"]).copy()
                if not dropped_active.empty:
                    dropped_active["Reason"] = "Contact has another active purchase deal"
                    show_removed_table(dropped_active, "Removed (active purchase on another deal)")

            # Exclude FUTURE td_booking_slot_date (active upcoming booking)
            today_mel = datetime.now(MEL_TZ).date()
            kept["slot_date_prop"] = kept["td_booking_slot_date"].apply(parse_epoch_or_iso_to_local_date)
            future_mask = kept["slot_date_prop"].apply(lambda d: isinstance(d, date) and d > today_mel)
            kept_no_future = kept[~future_mask].copy()
            if future_mask.any():
                future_rows = kept[future_mask].copy()
                future_rows["Reason"] = "Future TD booking date — likely upcoming appointment"
                show_removed_table(future_rows, "Removed (future bookings)")

            # 1) Filter internal/test emails + callout
            deals_f, removed_internal = filter_internal_test_emails(kept_no_future)

            # 2) Dedupe audit
            dedup, dedupe_dropped = dedupe_users_with_audit(deals_f, use_conducted=False)

            # 3) Messages audit
            msgs, skipped_msgs = build_messages_with_audit(dedup, mode="oldlead")

            # persist
            st.session_state["old_deals"] = deals_f
            st.session_state["old_removed_internal"] = removed_internal
            st.session_state["old_dedup"] = dedup
            st.session_state["old_dedupe_dropped"] = dedupe_dropped
            st.session_state["old_msgs"]  = msgs
            st.session_state["old_skipped_msgs"] = skipped_msgs

    deals_f      = st.session_state.get("old_deals")
    removed_int  = st.session_state.get("old_removed_internal")
    dedup        = st.session_state.get("old_dedup")
    dedupe_drop  = st.session_state.get("old_dedupe_dropped")
    msgs         = st.session_state.get("old_msgs")
    skipped_msgs = st.session_state.get("old_skipped_msgs")

    if isinstance(removed_int, pd.DataFrame) and not removed_int.empty:
        show_removed_table(removed_int, "Removed by domain filter (cars24.com / yopmail.com)")

    if isinstance(deals_f, pd.DataFrame) and not deals_f.empty:
        render_trimmed(deals_f, "Filtered deals (Old Leads — trimmed)", [
            ("full_name","Customer"), ("email","Email"),
            ("appointment_id","Appointment ID"),
            ("vehicle_make","Make"), ("vehicle_model","Model"),
            ("slot_date_prop","TD booking date"),
            ("conducted_date_local","TD conducted (date)"),
            ("dealstage","Stage"),
        ])

    if isinstance(dedupe_drop, pd.DataFrame) and not dedupe_drop.empty:
        show_removed_table(dedupe_drop, "Collapsed during dedupe (duplicates)")

    if isinstance(dedup, pd.DataFrame) and not dedup.empty:
        st.markdown("#### <span style='color:#000000;'>Deduped list (by mobile|email)</span>", unsafe_allow_html=True)
        st.dataframe(dedup[["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","DealStages"]]
                     .rename(columns={"WhenExact":"When (exact)","DealStages":"Stage(s)"}),
                     use_container_width=True)

    if isinstance(msgs, pd.DataFrame) and not msgs.empty:
        st.markdown("#### <span style='color:#000000;'>Message Preview (Old Leads)</span>", unsafe_allow_html=True)
        edited = render_selectable_messages(msgs, key="oldleads")
        if isinstance(skipped_msgs, pd.DataFrame) and not skipped_msgs.empty:
            st.markdown("**Skipped while creating SMS**")
            st.dataframe(skipped_msgs, use_container_width=True)

        if not edited.empty and st.button("Send SMS"):
            to_send = edited[edited["Send"]]
            if to_send.empty:
                st.warning("No rows selected.")
            elif not (AIRCALL_ID and AIRCALL_TOKEN and AIRCALL_NUMBER_ID):
                st.error("Missing Aircall credentials in .env.")
            else:
                st.info("Sending messages…")
                sent, failed = 0, 0
                for _, r in to_send.iterrows():
                    ok, msg = send_sms_via_aircall(r["Phone"], r["SMS draft"])
                    if ok: sent += 1; st.success(f"✅ Sent to {r['Phone']}")
                    else:  failed += 1; st.error(f"❌ Failed for {r['Phone']}: {msg}")
                    time.sleep(1)
                if sent: st.balloons()
                st.success(f"🎉 Done! Sent: {sent} | Failed: {failed}")


# ============ Router ============
if "view" not in st.session_state:
    st.session_state["view"]="home"

def header_and_route():
    header()
    v = st.session_state.get("view","home")
    if v == "home":
        ctas()
    elif v == "reminders":
        view_reminders()
    elif v == "manager":
        view_manager()
    elif v == "old":
        view_old()

header_and_route()
