# app.py
# Pawan Customer Connector — HubSpot + Aircall (Streamlit)
# Updated with working appointment_id-based car filtering

import os
import time
import json
from datetime import datetime, date, timezone, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv

# ============ Keys / Setup ============
load_dotenv()
HUBSPOT_TOKEN     = os.getenv("HUBSPOT_TOKEN", "")
AIRCALL_ID        = os.getenv("AIRCALL_ID")
AIRCALL_TOKEN     = os.getenv("AIRCALL_TOKEN")
AIRCALL_NUMBER_ID = os.getenv("AIRCALL_NUMBER_ID")
AIRCALL_NUMBER_ID_2 = os.getenv("AIRCALL_NUMBER_ID_2")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")

# Deployment timestamp
DEPLOYMENT_TIME = datetime.now(ZoneInfo("Australia/Melbourne")).strftime("%Y-%m-%d %H:%M:%S AEST")

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

# Pipeline & stages
PIPELINE_ID        = "2345821"
STAGE_ENQUIRY_ID   = "1119198251"  # Enquiry (no TD)
STAGE_BOOKED_ID    = "1119198252"  # 2. TD Booked
STAGE_CONDUCTED_ID = "1119198253"  # 3. TD Conducted (no deposit)

OLD_LEAD_START_STAGES = {STAGE_ENQUIRY_ID, STAGE_BOOKED_ID, STAGE_CONDUCTED_ID}

# Active purchase stages (exclude if any deal with same appointment_id has these stages)
ACTIVE_PURCHASE_STAGE_IDS = {
    "8082239", "8082240", "8082241", "8082242", "8082243", "8406593",
    "14816089", "14804235", "14804236", "14804237", "14804238",
    "14804239", "14804240"
}

DEAL_PROPS = [
    "hs_object_id", "dealname", "pipeline", "dealstage",
    "full_name", "email", "mobile", "phone",
    "appointment_id",
    "td_booking_slot", "td_booking_slot_date", "td_booking_slot_time",
    "td_conducted_date",
    "vehicle_make", "vehicle_model", "vehicle_year", "vehicle_colour", "vehicle_url",  
    "car_location_at_time_of_sale",
    "video_url__short_", 
    "td_reminder_sms_sent",
    "manager_followup_sms_sent",
]

STAGE_LABELS = {
    STAGE_ENQUIRY_ID:   "Enquiry (no TD)",
    STAGE_BOOKED_ID:    "TD booked",
    STAGE_CONDUCTED_ID: "TD conducted (no deposit)",
}

# ============ UI Theme ============
st.set_page_config(
    page_title="Pawan Customer Connector", 
    layout="wide",
    initial_sidebar_state="collapsed"
)
# Force light theme regardless of system settings
st._config.set_option('theme.base', 'light')

PRIMARY = "#4736FE"

st.markdown(f"""
<style>

/* ================================= */
/* MINIMAL DARK MODE OVERRIDE ONLY */
/* ================================= */

/* Override system dark mode detection */
:root {{
    color-scheme: light !important;
}}

[data-theme="dark"] {{
    color-scheme: light !important;
}}

.stApp[data-theme="dark"] {{
    background-color: #FFFFFF !important;
    color: #000000 !important;
}}

/* ================================= */
/* GLOBAL PAGE STYLING */
/* ================================= */

/* Force the entire app container to have white background */
html, body, [data-testid="stAppViewContainer"] {{
  background-color: #FFFFFF !important;  /* White background for entire page */
  color: #000000 !important;             /* Black text for entire page */
}}

/* Set maximum width for the main content area */
.block-container {{ 
  max-width: 1200px !important;          /* Limit content width to 1200px */
}}

/* ================================= */
/* HEADER STYLING */
/* ================================= */

/* Center the main page title */
.header-title {{ 
    color: {PRIMARY} !important;         /* Use primary blue color for title */
    text-align: center !important;       /* Center the title horizontally */
    margin: 0 !important;                /* Remove default margins */
}}

/* Style the horizontal divider line */
hr.div {{ 
  border: 0;                           /* Remove default border */
  border-top: 1px solid #E5E7EB;      /* Add thin gray top border */
  margin: 12px 0 8px;                  /* Add spacing above and below */
}}

/* ================================= */
/* BUTTON STYLING */
/* ================================= */

/* Style all Streamlit buttons */
div.stButton > button {{
    background-color: {PRIMARY} !important;  /* Blue background */
    color: #FFFFFF !important;               /* WHITE text on buttons */
    border: 1px solid {PRIMARY} !important;  /* Blue border */
    border-radius: 12px !important;          /* Rounded corners */
    font-weight: 600 !important;             /* Bold text */
}}

/* Button hover effects */
div.stButton > button:hover {{ 
    background-color: {PRIMARY} !important;  /* Keep blue on hover */
    color: #FFFFFF !important;               /* Keep WHITE text on hover */
}}

/* Special styling for call-to-action buttons */
div.stButton > button.cta {{ 
    width: 100% !important;                  /* Full width */
    height: 100px !important;                /* Taller height */
    font-size: 18px !important;              /* Larger text */
    text-align: left !important;             /* Left-align text */
    border-radius: 16px !important;          /* More rounded corners */
    color: #FFFFFF !important;               /* Ensure WHITE text */
}}

/* ================================= */
/* FORM STYLING */
/* ================================= */

/* Layout for form elements in a row */
.form-row {{ 
  display: flex !important;            /* Use flexbox layout */
  justify-content: center !important;  /* Center horizontally */
  align-items: end !important;         /* Align to bottom */
  gap: 12px !important;                /* Space between elements */
  flex-wrap: wrap !important;          /* Wrap on small screens */
}}

/* Style all form inputs */
input, select, textarea {{
  background-color: #FFFFFF !important;  /* White background for inputs */
  color: #000000 !important;             /* Black text in inputs */
  border: 1px solid #D1D5DB !important;  /* Gray border */
  border-radius: 10px !important;        /* Rounded corners */
}}

/* Style all form labels */
label, .stSelectbox label, .stDateInput label, .stTextInput label {{ 
  color: #000000 !important;           /* Black text for labels */
}}

/* ================================= */
/* TABLE STYLING - AGGRESSIVE APPROACH */
/* ================================= */

/* ATTEMPT 1: Target main dataframe container */
[data-testid="stDataFrame"] {{
    background-color: #FFFFFF !important;  /* Force white background */
    color: #000000 !important;             /* Force black text */
}}

/* ATTEMPT 2: Target ALL children of dataframe */
[data-testid="stDataFrame"] * {{
    color: #000000 !important;             /* Force ALL child elements to black text */
    background-color: transparent !important; /* Transparent background for children */
}}

/* ATTEMPT 3: Target specific table cell types */
[data-testid="stDataFrame"] div[role="cell"] {{
  background-color: #FFFFFF !important;    /* White background for cells */
  color: #000000 !important;               /* BLACK text for cells */
  border: 1px solid #CCCCCC !important;    /* Gray border to see cell boundaries */
  padding: 8px !important;                 /* Padding inside cells */
}}

/* ATTEMPT 4: Target column headers specifically */
[data-testid="stDataFrame"] div[role="columnheader"] {{
  background-color: #F8F9FA !important;    /* Light gray background for headers */
  color: #000000 !important;               /* BLACK text for headers */
  font-weight: bold !important;            /* Bold header text */
  border: 1px solid #CCCCCC !important;    /* Gray border */
  padding: 8px !important;                 /* Padding inside headers */
}}

/* ATTEMPT 5: Target grid cells */
[data-testid="stDataFrame"] div[role="gridcell"] {{
  background-color: #FFFFFF !important;    /* White background */
  color: #000000 !important;               /* BLACK text */
  border: 1px solid #CCCCCC !important;    /* Gray border */
  padding: 8px !important;                 /* Padding */
}}

/* ATTEMPT 6: Target any div inside table cells */
[data-testid="stDataFrame"] div[role="cell"] div {{
    color: #000000 !important;             /* Force black text on cell divs */
}}

[data-testid="stDataFrame"] div[role="columnheader"] div {{
    color: #000000 !important;             /* Force black text on header divs */
}}

[data-testid="stDataFrame"] div[role="gridcell"] div {{
    color: #000000 !important;             /* Force black text on gridcell divs */
}}

/* ATTEMPT 7: Target spans inside cells */
[data-testid="stDataFrame"] span {{
    color: #000000 !important;             /* Force black text on spans */
}}

/* ATTEMPT 8: Alternative dataframe selectors */
.stDataFrame {{
    color: #000000 !important;             /* Black text for stDataFrame class */
}}

.stDataFrame * {{
    color: #000000 !important;             /* Black text for all children */
}}

/* ATTEMPT 9: Use CSS pseudo-selectors */
[data-testid="stDataFrame"] *:not(button):not(input) {{
    color: #000000 !important;             /* Black text except buttons and inputs */
}}

/* ATTEMPT 10: Nuclear option - override any text color */
div[data-testid="stDataFrame"] {{
    color: #000000 !important;
}}

div[data-testid="stDataFrame"] > * {{
    color: #000000 !important;
}}

div[data-testid="stDataFrame"] > * > * {{
    color: #000000 !important;
}}

div[data-testid="stDataFrame"] > * > * > * {{
    color: #000000 !important;
}}

/* ================================= */
/* ALTERNATIVE TABLE STYLING */
/* ================================= */

/* Style regular HTML tables if Streamlit falls back to them */
table {{
    background-color: #FFFFFF !important;  /* White table background */
    color: #000000 !important;             /* Black table text */
    border-collapse: collapse !important;   /* Merge borders */
}}

table td, table th {{
    background-color: #FFFFFF !important;  /* White cell background */
    color: #000000 !important;             /* BLACK cell text */
    border: 1px solid #CCCCCC !important;  /* Gray cell borders */
    padding: 8px !important;               /* Cell padding */
}}

/* ================================= */
/* TEXT WRAPPING */
/* ================================= */

/* Ensure text wraps properly in all table cells */
[data-testid="stDataFrame"] div[role="cell"],
[data-testid="stDataFrame"] div[role="columnheader"],
[data-testid="stDataFrame"] div[role="gridcell"] {{
  white-space: pre-wrap !important;        /* Preserve line breaks and wrap */
  word-wrap: break-word !important;        /* Break long words */
  overflow-wrap: anywhere !important;      /* Allow breaking anywhere */
  line-height: 1.4 !important;             /* Readable line spacing */
  max-width: none !important;              /* No width restrictions */
  height: auto !important;                 /* Auto height */
  min-height: 40px !important;             /* Minimum cell height */
  vertical-align: top !important;          /* Align content to top */
  overflow: visible !important;            /* Show all content */
}}
</style>

<script>
/* ================================= */
/* JAVASCRIPT BACKUP APPROACH */
/* ================================= */

/* If CSS fails, use JavaScript to force black text */
setTimeout(function() {{
    /* Find all dataframe elements */
    var dataframes = document.querySelectorAll('[data-testid="stDataFrame"]');
    
    /* Loop through each dataframe */
    dataframes.forEach(function(df) {{
        /* Set black text on the container */
        df.style.color = '#000000';
        df.style.backgroundColor = '#FFFFFF';
        
        /* Find all child elements and force black text */
        var allChildren = df.querySelectorAll('*');
        allChildren.forEach(function(child) {{
            child.style.color = '#000000';
            /* Don't override button backgrounds */
            if (!child.matches('button')) {{
                child.style.backgroundColor = 'transparent';
            }}
        }});
    }});
    
    /* Log to console for debugging */
    console.log('Applied black text to', dataframes.length, 'dataframes');
}}, 1000); /* Wait 1 second for page to load */
</script>
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

def force_light_theme():
    """Force light theme regardless of system settings"""
    st.markdown("""
    <script>
    // Override system preference
    window.matchMedia = function(query) {
        if (query === '(prefers-color-scheme: dark)') {
            return {
                matches: false,
                addListener: function() {},
                removeListener: function() {}
            };
        }
        return originalMatchMedia(query);
    };
    </script>
    """, unsafe_allow_html=True)
def simplify_vehicle_color(color_name: str) -> str:
    """Simplify complex manufacturer color names to basic colors for SMS messages"""
    if not color_name or pd.isna(color_name):
        return ""
    
    color = str(color_name).lower().strip()
    
    # Red variations
    if any(word in color for word in ['red', 'crimson', 'scarlet', 'burgundy', 'ruby', 'cherry', 'rose']):
        return "Red"
    # Blue variations  
    if any(word in color for word in ['blue', 'navy', 'azure', 'cobalt', 'sapphire', 'indigo', 'teal']):
        return "Blue"
    # White variations
    if any(word in color for word in ['white', 'pearl', 'ivory', 'cream', 'snow', 'frost']):
        return "White"
    # Black variations
    if any(word in color for word in ['black', 'ebony', 'coal', 'charcoal', 'onyx', 'midnight']):
        return "Black"
    # Silver/Grey variations
    if any(word in color for word in ['silver', 'grey', 'gray', 'platinum', 'steel', 'graphite', 'titanium']):
        return "Silver"
    # Green variations
    if any(word in color for word in ['green', 'emerald', 'forest', 'sage', 'olive', 'lime']):
        return "Green"
    # Yellow/Gold variations
    if any(word in color for word in ['yellow', 'gold', 'amber', 'champagne', 'bronze']):
        return "Gold"
    # Orange variations
    if any(word in color for word in ['orange', 'copper', 'sunset', 'rust']):
        return "Orange"
    # Purple variations
    if any(word in color for word in ['purple', 'violet', 'magenta', 'plum']):
        return "Purple"
    # Brown variations
    if any(word in color for word in ['brown', 'tan', 'beige', 'mocha', 'coffee', 'chocolate']):
        return "Brown"
    
    # If no match found, try to extract the first recognizable color word
    color_words = color.split()
    for word in color_words:
        if word in ['red', 'blue', 'white', 'black', 'silver', 'green', 'yellow', 'orange', 'purple', 'brown']:
            return word.capitalize()
    
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
    df["slot_time"]      = df["td_booking_slot"].apply(parse_epoch_or_iso_to_local_time)
    df["slot_date_prop"] = df["td_booking_slot_date"].apply(parse_epoch_or_iso_to_local_date)
    df["slot_time_param"]= df["td_booking_slot_time"].apply(parse_td_slot_time_prop)
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

def filter_sms_already_sent(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out deals where td_reminder_sms_sent is already 'true' or 'Yes'.
    Returns (kept_df, removed_df) where removed_df includes a Reason column.
    """
    if df is None or df.empty or "td_reminder_sms_sent" not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(), pd.DataFrame()
    
    work = df.copy()
    
    # Check if SMS was already sent - check for both "true" (value) and "Yes" (label)
    work["sms_sent"] = work["td_reminder_sms_sent"].apply(
        lambda x: str(x).lower() in ['yes', 'true'] if pd.notna(x) else False
    )
    
    removed = work[work["sms_sent"]].drop(columns=["sms_sent"]).copy()
    kept = work[~work["sms_sent"]].drop(columns=["sms_sent"]).copy()
    
    if not removed.empty:
        removed["Reason"] = "SMS reminder already sent (td_reminder_sms_sent = true)"
    
    return kept, removed

def filter_manager_sms_already_sent(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out deals where manager_followup_sms_sent is already 'true' or 'Yes'.
    Returns (kept_df, removed_df) where removed_df includes a Reason column.
    """
    if df is None or df.empty or "manager_followup_sms_sent" not in df.columns:
        return df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame(), pd.DataFrame()

    work = df.copy()

    # Treat both 'true' and 'Yes' as already sent (value vs label)
    work["manager_sms_sent_flag"] = work["manager_followup_sms_sent"].apply(
        lambda x: str(x).lower() in ['yes', 'true'] if pd.notna(x) else False
    )

    removed = work[work["manager_sms_sent_flag"]].drop(columns=["manager_sms_sent_flag"]).copy()
    kept    = work[~work["manager_sms_sent_flag"]].drop(columns=["manager_sms_sent_flag"]).copy()

    if not removed.empty:
        removed["Reason"] = "Manager follow-up already sent"

    return kept, removed

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

def update_deals_manager_followup_sms_sent(deal_ids: list[str]) -> tuple[int, int]:
    '''
    Update the manager_followup_sms_sent property to 'true' for the given deal IDs.
    Returns (success_count, failure_count)
    '''
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
                    "manager_followup_sms_sent": "true"
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


def get_all_deal_ids_for_contacts(messages_df: pd.DataFrame, deals_df: pd.DataFrame) -> dict[str, list[str]]:
    """
    For each phone number in messages_df, get all associated deal IDs from deals_df.
    Returns a dict mapping phone -> list of deal IDs
    """
    phone_to_deals = {}
    
    if messages_df is None or messages_df.empty or deals_df is None or deals_df.empty:
        return phone_to_deals
    
    # Create a mapping of normalized phone to deal IDs
    for _, msg_row in messages_df.iterrows():
        phone = str(msg_row.get("Phone", "")).strip()
        if not phone:
            continue
        
        # Find all deals with this phone number
        matching_deals = deals_df[deals_df["phone_norm"] == phone]["hs_object_id"].tolist()
        phone_to_deals[phone] = [str(d) for d in matching_deals if d]
    
    return phone_to_deals

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

def build_messages_with_audit(dedup_df: pd.DataFrame, mode: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build messages; also return a 'skipped' DF with reasons (e.g., missing phone, empty draft).
    """
    msgs_df = build_messages_from_dedup(dedup_df, mode=mode)
    skipped = []
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
                if msgs_df.empty or not (msgs_df["Phone"] == phone).any():
                    skipped.append({
                        "Customer": str(row.get("CustomerName") or ""),
                        "Email": str(row.get("Email") or ""),
                        "Cars": str(row.get("Cars") or ""),
                        "Reason": "No message generated"
                    })
    skipped_df = pd.DataFrame(skipped, columns=["Customer","Email","Cars","Reason"])
    return msgs_df, skipped_df

def fix_json_response(response_text):
    """Try to fix common JSON formatting issues from ChatGPT"""
    try:
        # Remove any text before the first {
        start_idx = response_text.find('{')
        if start_idx > 0:
            response_text = response_text[start_idx:]
        
        # Remove any text after the last }
        end_idx = response_text.rfind('}')
        if end_idx > 0:
            response_text = response_text[:end_idx + 1]
        
        # Fix common escape issues
        response_text = response_text.replace('\n', '\\n').replace('\t', '\\t')
        
        # Try to parse and return if successful
        json.loads(response_text)
        return response_text
    except:
        return None

def create_fallback_analysis(raw_response, customer_name):
    """Create a structured response when JSON parsing fails"""
    lines = raw_response.split('\n')
    
    summary = "Analysis incomplete due to formatting issues"
    category = "No clear reason documented"
    next_steps = "Review notes manually and contact customer"
    
    # Try to extract summary from response
    for line in lines:
        if any(word in line.lower() for word in ['summary', 'what happened', 'customer']):
            if len(line.strip()) > 10:
                summary = line.strip()[:100]
                break
    
    return {
        "summary": summary,
        "category": category,
        "next_steps": next_steps,
        "raw_response": raw_response[:200] + "..." if len(raw_response) > 200 else raw_response
    }
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

def get_contact_note_ids(contact_id):
    """Get note IDs for a contact"""
    headers = {"Authorization": f"Bearer {HUBSPOT_TOKEN}"}
    try:
        url = f"{HS_ROOT}/crm/v3/objects/contacts/{contact_id}/associations/notes"
        response = requests.get(url, headers=headers, timeout=25)
        
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            note_ids = [result.get("toObjectId") or result.get("id") for result in results]
            return [str(nid) for nid in note_ids if nid]
        else:
            return []
    except Exception:
        return []

def get_notes_content(note_ids):
    """Get note content"""
    headers = {"Authorization": f"Bearer {HUBSPOT_TOKEN}"}
    if not note_ids:
        return []
    
    try:
        url = f"{HS_ROOT}/crm/v3/objects/notes/batch/read"
        payload = {
            "properties": ["hs_note_body", "hs_timestamp", "hs_createdate", "hubspot_owner_id"],
            "inputs": [{"id": str(note_id)} for note_id in note_ids]
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=25)
        
        if response.status_code == 200:
            data = response.json()
            return data.get("results", [])
        else:
            return []
    except:
        return []

def get_owner_name(owner_id):
    """Get owner name"""
    headers = {"Authorization": f"Bearer {HUBSPOT_TOKEN}"}
    if not owner_id:
        return "Unknown User"
    
    try:
        url = f"{HS_ROOT}/crm/v3/owners/{owner_id}"
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            first_name = data.get("firstName", "")
            last_name = data.get("lastName", "")
            if first_name or last_name:
                return f"{first_name} {last_name}".strip()
            else:
                return data.get("email", f"User {owner_id}")
        else:
            return f"User {owner_id}"
    except:
        return f"User {owner_id}"

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


def analyze_with_chatgpt(notes_text, customer_name="Customer", vehicle="Vehicle"):
    """Analyze customer notes using ChatGPT with enhanced debugging"""
    if not notes_text or notes_text == "No notes":
        return {
            "summary": "No notes available for analysis",
            "category": "No clear reason documented",
            "next_steps": "Contact customer to understand their experience"
        }
    
    system_prompt = """You are analyzing customer interaction notes from a car dealership to understand why customers didn't pay a deposit after test drives.

CRITICAL: You must respond with ONLY valid JSON in exactly this format - no extra text, no explanations, just the JSON:

{
  "summary": "1-2 line summary of what specifically happened during customer interaction and why deposit was not paid",
  "category": "choose one category from the list below", 
  "next_steps": "specific actionable next step for the sales team to re-engage this customer"
}

Categories (choose exactly one):
- Price/Finance Issues
- Vehicle Condition/Quality  
- Customer Not Ready
- Comparison Shopping
- Feature/Specification Issues
- Trust/Service Issues
- External Factors
- Already Purchased Elsewhere
- Changed Mind/Lost Interest
- No clear reason documented

Rules:
- Response must be valid JSON only
- Keep summary under 150 characters
- Keep next_steps under 100 characters
- Use only the categories listed above exactly as written"""

    user_prompt = f"""Customer: {customer_name}
Vehicle: {vehicle}

Customer interaction notes from dealership:
{notes_text}

Analyze why this customer didn't pay a deposit after their test drive and what the sales team should do next."""

    try:
        import openai
        
        # Set API key
        openai_api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")
        if not openai_api_key:
            return {
                "summary": "OpenAI API key not configured",
                "category": "Analysis failed",
                "next_steps": "Configure OpenAI API key in secrets"
            }

        openai.api_key = openai_api_key
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=250
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Try to parse JSON
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as json_err:
            # Try to fix common JSON issues
            fixed_response = fix_json_response(response_text)
            if fixed_response:
                try:
                    result = json.loads(fixed_response)
                except:
                    return create_fallback_analysis(response_text, customer_name)
            else:
                return create_fallback_analysis(response_text, customer_name)
        
        return {
            "summary": result.get("summary", "Analysis incomplete"),
            "category": result.get("category", "No clear reason documented"),
            "next_steps": result.get("next_steps", "Review customer interaction manually")
        }
        
    except Exception as e:
        return {
            "summary": f"ChatGPT analysis failed: {str(e)[:50]}...",
            "category": "Analysis failed",
            "next_steps": "Review notes manually and contact customer"
        }

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

def filter_deals_by_appointment_id_car_active_purchases(deals_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter out deals where other deals with the same appointment_id have active purchase stages.
    This works by:
    1. Getting the appointment_id for each deal
    2. Finding all other deals with the same appointment_id  
    3. Checking if any of those other deals have active purchase stages
    4. Excluding the original deal if so
    """
    if deals_df is None or deals_df.empty:
        return deals_df.copy() if isinstance(deals_df, pd.DataFrame) else pd.DataFrame(), pd.DataFrame()
    
    # Get deal IDs
    deal_ids = deals_df.get("hs_object_id", pd.Series(dtype=str)).dropna().astype(str).tolist()
    if not deal_ids:
        return deals_df.copy(), pd.DataFrame()
    
    # Get appointment_id for each deal
    deal_appointment_map = {}
    deal_data = hs_batch_read_deals(deal_ids, props=["appointment_id"])
    
    for deal_id in deal_ids:
        props = deal_data.get(deal_id, {})
        appointment_id = props.get("appointment_id")
        if appointment_id:
            deal_appointment_map[deal_id] = str(appointment_id).strip()
    
    if not deal_appointment_map:
        return deals_df.copy(), pd.DataFrame()
    
    # For each unique appointment_id, find all deals with that appointment_id
    appointment_ids = set(deal_appointment_map.values())
    
    # Get all deals for each appointment_id and check their stages
    deals_to_exclude = set()
    
    for appointment_id in appointment_ids:
        # Get all deals with this appointment_id
        all_deals_for_appointment = get_deals_by_appointment_id(appointment_id)
        
        # Get stages for all these deals
        if all_deals_for_appointment:
            stage_data = hs_batch_read_deals(all_deals_for_appointment, props=["dealstage"])
            
            # Check if any deal (other than our original ones) has active purchase stage
            has_active_purchase = False
            for check_deal_id in all_deals_for_appointment:
                if check_deal_id in deal_ids:
                    continue  # Skip our original deals
                
                stage = (stage_data.get(check_deal_id, {}) or {}).get("dealstage")
                
                if stage and str(stage) in ACTIVE_PURCHASE_STAGE_IDS:
                    has_active_purchase = True
                    break
            
            # If any other deal has active purchase stage, exclude all our original deals with this appointment_id
            if has_active_purchase:
                for deal_id, deal_appointment in deal_appointment_map.items():
                    if deal_appointment == appointment_id:
                        deals_to_exclude.add(deal_id)
    
    # Filter the dataframe
    work = deals_df.copy()
    work["__keep"] = work["hs_object_id"].apply(lambda x: str(x) not in deals_to_exclude)
    
    dropped = work[~work["__keep"]].drop(columns=["__keep"]).copy()
    kept = work[work["__keep"]].drop(columns=["__keep"]).copy()
    
    if not dropped.empty:
        dropped["Reason"] = "Car (via appointment_id) has another deal in active purchase stage"
    
    return kept, dropped

# ============ HubSpot ============
@st.cache_data(show_spinner=False)
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
def send_sms_via_aircall(phone: str, message: str, number_id: str = None) -> tuple[bool, str]:
    """Send SMS with specified Aircall number ID"""
    # Use default number if none specified OR if empty string
    if not number_id:  # This catches None, "", and other falsy values
        number_id = AIRCALL_NUMBER_ID
    
    try:
        url = f"{AIRCALL_BASE_URL}/numbers/{number_id}/messages/native/send"
        print(f"DEBUG: Using Aircall number ID: {number_id}")  # Debug line
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

def draft_sms_reminder(name: str, pairs_text: str, video_urls: str = "") -> str:
    system = (
        "You write outbound SMS for Cars24 Laverton (Australia). "
        "Tone: warm, polite, inviting, Australian. AU spelling. "
        "Keep ~280 chars unless you are going to add video URLs too. can add limited emojis. Avoid apostrophes."
        "Write as the business (sender). Include a clear CTA to confirm or reschedule."
    )
    # Check if video URLs are provided
    video_url_list = [url.strip() for url in video_urls.split(";") if url.strip()] if video_urls else []
    has_video = len(video_url_list) > 0
    
    if has_video:
        # Use first video URL and allow longer message
        first_video_url = video_url_list[0]
        system += " If video URL provided, encourage virtual tour before test drive. Keep ~400 chars max. No emojis/links except the provided video URL. Avoid apostrophes."
        user = f"Recipient name: {name or 'there'}.\nUpcoming test drive(s): {pairs_text}.\nVideo tour URL: {first_video_url}.\nFriendly reminder with video tour suggestion."
    else:
        # Standard message without video
        system += " Keep ~280 chars. No emojis/links. Avoid apostrophes."
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
        "Keep ~400 characters. No emojis/links. Avoid apostrophes."
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

def draft_sms_oldlead_by_stage_improved(name: str, vehicle_details: list, stage_hint: str) -> str:
    """Generate improved SMS for old leads with vehicle details and stage-specific messaging using ChatGPT"""
    first = (name or "").split()[0] if (name or "").strip() else "there"
    
    # Use first vehicle for primary messaging
    primary_vehicle = vehicle_details[0] if vehicle_details else {}
    make = primary_vehicle.get('make', '')
    model = primary_vehicle.get('model', '')
    year = primary_vehicle.get('year', '')
    color = primary_vehicle.get('color', '')
    url = primary_vehicle.get('url', '')
    stage_id = primary_vehicle.get('stage_id', '')
    
    # Build vehicle description for ChatGPT
    vehicle_parts = []
    if year: vehicle_parts.append(year)
    if color: vehicle_parts.append(color)
    if make: vehicle_parts.append(make)
    if model: vehicle_parts.append(model)
    
    vehicle_text = " ".join(vehicle_parts) if vehicle_parts else "the vehicle"
    
    # Stage-specific context and messaging based on stage IDs
    if stage_id == "1119198251" or stage_hint == "enquiry":  # Enquiry stage
        context = "They enquired but have not booked a test drive yet."
        ask = "Ask if they are still looking for a car and encourage booking a test drive to meet in person when they are on site."
        stage_specific_action = "Are you still looking for a car? I would love to meet you in person when you are on site for a test drive."
    elif stage_id == "1119198252" or stage_hint == "booked":  # TD Booked stage  
        context = "They booked a test drive but did not show up."
        ask = "Encourage them to drive down to Laverton, mention the drive would be worth it, ask about change of plans."
        stage_specific_action = "I encourage you to drive down to Laverton - the drive would definitely be worth it! Has there been any change of plans?"
    elif stage_id == "1119198253" or stage_hint == "conducted":  # TD Conducted stage
        context = "They completed a test drive but did not proceed with purchase."
        ask = "Check what could be done differently to make this work for you."
        stage_specific_action = "Is there anything I could do differently to make this work for you?"
    else:
        context = "It has been a while since they reached out."
        ask = "Re-engage and check current interest."
        stage_specific_action = "Are you still in the market for a vehicle? I am here to help find the perfect deal."
    
    # Enhanced system prompt for ChatGPT
    system = (
        "You write outbound SMS for Cars24 Laverton (Australia) from the store manager, Pawan. "
        "Tone: warm, courteous, Australian; avoid pressure; encourage a reply. "
        "Promise personal attention and that we will work out a deal they will love. "
        "Include the vehicle URL if provided so customer can identify the specific car. "
        "Keep ~300 characters if no URL, or ~400 characters if URL included. "
        "No emojis/links except the provided vehicle URL. Avoid apostrophes."
    )
    
    # Enhanced user prompt with vehicle details
    user = (
        f"Recipient name: {name or 'there'}.\n"
        f"Vehicle of interest: {vehicle_text}\n"
        f"Vehicle URL (include if provided): {url}\n"
        f"Stage context: {context}\n"
        f"Suggested stage-specific action: {stage_specific_action}\n"
        f"Begin the SMS with exactly: Hi {first}, this is Pawan, Sales Manager at Cars24 Laverton.\n"
        f"{ask} Include the vehicle URL in the message if provided. Make it friendly and concise."
    )
    
    # Call ChatGPT
    text = _call_openai([
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]) or ""
    
    # Fallback if ChatGPT fails
    if not text.strip():
        vehicle_url_text = f" {url}" if url else ""
        text = f"Hi {first}, this is Pawan, Sales Manager at Cars24 Laverton. Hope you are well! Regarding the {vehicle_text}{vehicle_url_text} - {stage_specific_action} Please let me know. Thanks!"
    
    # Ensure proper ending format
    intro = f"hi {first.lower()}, this is pawan, sales manager at cars24 laverton"
    if text.strip().lower().startswith(intro): 
        return text.strip()
    return f"{text.strip()} –Pawan, Sales Manager"

# ============ Dedupe & SMS build ============
def dedupe_users(df: pd.DataFrame, *, use_conducted: bool) -> pd.DataFrame:
    """Return rows with: CustomerName, Phone, Email, DealsCount, Cars, WhenExact, WhenRel, DealStages, StageHint, VehicleDetails."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","WhenRel","DealStages","StageHint","VehicleDetails"])
    work = df.copy()
    work["email_l"] = work["email"].astype(str).str.strip().str.lower()
    work["user_key"] = (work["phone_norm"].fillna('') + "|" + work["email_l"].fillna('')).str.strip()
    work = work[work["user_key"].astype(bool)]
    rows = []
    for _, grp in work.groupby("user_key", sort=False):
        name  = first_nonempty_str(grp["full_name"])
        phone = first_nonempty_str(grp["phone_norm"])
        email = first_nonempty_str(grp["email"])
        cars_list, when_exact_list, when_rel_list, stages_list, video_urls_list = [], [], [], [], []
        vehicle_details_list = []  # NEW: Store detailed vehicle info
        
        for _, r in grp.iterrows():
            # Build basic car description
            car = f"{str(r.get('vehicle_make') or '').strip()} {str(r.get('vehicle_model') or '').strip()}".strip() or "car"
            
            # NEW: Build detailed vehicle info for messaging
            vehicle_year = str(r.get('vehicle_year') or '').strip()
            vehicle_colour = str(r.get('vehicle_colour') or '').strip()
            vehicle_url = str(r.get('vehicle_url') or '').strip()
            simplified_color = simplify_vehicle_color(vehicle_colour)
            stage_id = str(r.get('dealstage') or '').strip()
            
            # Store detailed vehicle info as dict
            vehicle_detail = {
                'make': str(r.get('vehicle_make') or '').strip(),
                'model': str(r.get('vehicle_model') or '').strip(), 
                'year': vehicle_year,
                'color': simplified_color,
                'url': vehicle_url,
                'stage_id': stage_id
            }
            vehicle_details_list.append(vehicle_detail)
            
            if use_conducted:
                d = r.get("conducted_date_local"); t = r.get("conducted_time_local") or ""
            else:
                d = r.get("slot_date_prop") or r.get("slot_date")
                t = r.get("slot_time_param") or r.get("slot_time") or ""
            when_rel = rel_date(d) if isinstance(d, date) else ""
            when_exact = (f"{format_date_au(d)} {t}".strip()).strip()

            # Collect video URL
            video_url = str(r.get("video_url__short_") or "").strip()

            cars_list.append(car)
            when_exact_list.append(when_exact)
            when_rel_list.append(when_rel if t == "" else f"{when_rel} at {t}".strip())
            stages_list.append(str(r.get("dealstage") or ""))
            if video_url:  # Only add non-empty video URLs
                video_urls_list.append(video_url)

        stage_labels = sorted({stage_label(x) for x in stages_list if str(x)})
        if STAGE_CONDUCTED_ID in stages_list: hint = "conducted"
        elif STAGE_BOOKED_ID in stages_list: hint = "booked"
        elif STAGE_ENQUIRY_ID in stages_list: hint = "enquiry"
        else: hint = "unknown"

        # Get unique video URLs
        unique_video_urls = list(set([url for url in video_urls_list if url]))

        rows.append({
            "CustomerName": name, "Phone": phone, "Email": email,
            "DealsCount": len([c for c in cars_list if c]),
            "Cars": "; ".join([c for c in cars_list if c]),
            "WhenExact": "; ".join([w for w in when_exact_list if w]),
            "WhenRel": "; ".join([w for w in when_rel_list if w]),
            "DealStages": "; ".join(stage_labels) if stage_labels else "",
            "StageHint": hint,
            "VideoURLs": "; ".join(unique_video_urls) if unique_video_urls else "",
            "VehicleDetails": vehicle_details_list  # NEW: Detailed vehicle info
        })
    out = pd.DataFrame(rows)
    want = ["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","WhenRel","DealStages","StageHint","VideoURLs","VehicleDetails"]
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
        video_urls = str(row.get("VideoURLs") or "").strip()
        vehicle_details = row.get("VehicleDetails", [])  # NEW: Get vehicle details
        
        pairs_text = build_pairs_text(cars, when_rel)
        
        if mode == "reminder":
            msg = draft_sms_reminder(name, pairs_text, video_urls)
        elif mode == "manager":
            msg = draft_sms_manager(name, pairs_text)
        elif mode == "oldlead":
            # Use improved old lead messaging
            stage_hint = str(row.get("StageHint") or "unknown")
            msg = draft_sms_oldlead_by_stage_improved(name, vehicle_details, stage_hint)
        else:
            # Fallback to original for other modes
            car_text = cars or "the car you were eyeing"
            stage_hint = str(row.get("StageHint") or "unknown")
            msg = draft_sms_oldlead_by_stage(name, car_text, stage_hint)
            
        out.append({"CustomerName": name, "Phone": phone, "Email": str(row.get("Email") or "").strip(),
                    "Cars": cars, "WhenExact": str(row.get("WhenExact") or ""), "WhenRel": when_rel,
                    "DealStages": str(row.get("DealStages") or ""), "Message": msg})
    return pd.DataFrame(out, columns=["CustomerName","Phone","Email","Cars","WhenExact","WhenRel","DealStages","Message"])

def view_unsold_summary():
    st.subheader("📊  Unsold TD Summary")
    
    with st.form("unsold_summary_form"):
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        c1,c2,c3,c4 = st.columns([1.4,1.6,1.6,2.0])
        
        with c1: 
            mode = st.radio("Mode", ["Single date","Date range"], horizontal=True, index=1)
        
        today = datetime.now(MEL_TZ).date()
        if mode=="Single date":
            with c2: d1 = st.date_input("Date", value=today); d2 = d1
            with c3: pass
        else:
            with c2: d1 = st.date_input("Start date", value=today - timedelta(days=7))
            with c3: d2 = st.date_input("End date", value=today)
        
        # State filter
        state_options = hs_get_deal_property_options("car_location_at_time_of_sale")
        values = [o["value"] for o in state_options] if state_options else []
        labels = [o["label"] for o in state_options] if state_options else []
        def_val = "VIC" if "VIC" in values else (values[0] if values else "")
        
        with c4:
            if labels:
                chosen_label = st.selectbox("Vehicle state", labels, index=(values.index("VIC") if "VIC" in values else 0))
                label_to_val = {o["label"]:o["value"] for o in state_options}
                state_val = label_to_val.get(chosen_label, def_val)
            else:
                state_val = st.text_input("Vehicle state", value=def_val)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Ticket Owner selection (simplified)
        st.markdown("**Ticket Owner:** (All owners for now)")
        
        go = st.form_submit_button("Analyze Unsold TDs", use_container_width=True)
    
    if go:
        with st.spinner("Fetching and analyzing deals..."):
            try:
                # Get deals
                start_ms, _ = mel_day_bounds_to_epoch_ms(d1)
                _, end_ms = mel_day_bounds_to_epoch_ms(d2)
                
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
                    st.info("No deals found matching the criteria.")
                    return
                
                deals_df = prepare_deals(raw_deals)
                
                # Dedupe deals by customer (email/phone combination)
                deals_df["email_l"] = deals_df["email"].astype(str).str.strip().str.lower()
                deals_df["user_key"] = (deals_df["phone_norm"].fillna('') + "|" + deals_df["email_l"].fillna('')).str.strip()
                deals_df = deals_df[deals_df["user_key"].astype(bool)]
                
                # Keep first deal per customer and collect all vehicles
                dedupe_results = []
                for user_key, group in deals_df.groupby("user_key"):
                    # Take first deal as primary
                    primary_deal = group.iloc[0]
                    
                    # Collect all vehicles and appointment_ids for this customer
                    vehicles_info = []
                    for _, deal_row in group.iterrows():
                        vehicle_make = deal_row.get('vehicle_make', '')
                        vehicle_model = deal_row.get('vehicle_model', '')
                        appointment_id = deal_row.get('appointment_id', '')
                        vehicle_info = f"{vehicle_make} {vehicle_model}".strip()
                        if appointment_id:
                            vehicle_info += f" (ID: {appointment_id})"
                        vehicles_info.append(vehicle_info)
                    
                    # Create combined record
                    combined_deal = primary_deal.copy()
                    combined_deal['all_vehicles'] = " | ".join(vehicles_info)
                    combined_deal['deal_count'] = len(group)
                    dedupe_results.append(combined_deal)
                
                deals_df = pd.DataFrame(dedupe_results)
                
                if deals_df.empty:
                    st.info("No deals found after processing.")
                    return
                
                # Process each deal
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, (_, deal_row) in enumerate(deals_df.iterrows()):
                    deal_id = str(deal_row.get('hs_object_id', 'Unknown'))
                    customer_name = str(deal_row.get('full_name', 'Unknown Customer'))
                    vehicle = f"{deal_row.get('vehicle_make', '')} {deal_row.get('vehicle_model', '')}".strip() or "Unknown Vehicle"
                    
                    status_text.text(f"Processing {i+1}/{len(deals_df)}: {customer_name}")
                    progress_bar.progress((i + 1) / len(deals_df))
                    
                    # Get consolidated notes
                    try:
                        notes = get_consolidated_notes_for_deal(deal_id)
                        if not notes or notes.strip() == "" or notes == "No notes":
                            notes = "No notes"
                    except Exception as e:
                        notes = f"Error getting notes: {str(e)}"
                    
                    # Analyze with ChatGPT
                    try:
                        analysis = analyze_with_chatgpt(notes, customer_name, vehicle)
                    except Exception as e:
                        analysis = {
                            "summary": f"Analysis failed: {str(e)[:50]}...",
                            "category": "Analysis failed",
                            "next_steps": "Review manually"
                        }
                    
                    # Format notes for display with line breaks
                    display_notes = notes[:300] + "..." if len(notes) > 300 else notes
                    display_notes = display_notes.replace('\n\n', '\n').replace('\n', ' | ')
                    
                    results.append({
                        "Deal ID": deal_id,
                        "Customer": customer_name,
                        "Vehicle": deal_row.get('all_vehicles', vehicle),  # Use combined vehicles
                        "Notes": display_notes,
                        "Summary": analysis.get("summary", "No summary"),
                        "Category": analysis.get("category", "Unknown"),
                        "Next Steps": analysis.get("next_steps", "No steps"),
                        "Deal Count": deal_row.get('deal_count', 1),
                        "TD Date": deal_row.get('conducted_date_local', 'Unknown')  # Add date for weekly breakdown
                    })
                
                progress_bar.empty()
                status_text.empty()
                
                # Store results
                st.session_state["unsold_results"] = results
                st.success(f"Successfully analyzed {len(results)} deals!")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                return
    
    # Display results
    results = st.session_state.get("unsold_results")
    if results:
        st.markdown(f"#### Unsold Test Drive Analysis ({len(results)} deals)")
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Display with specific column widths and configurations
        st.dataframe(
            results_df, 
            use_container_width=True, 
            hide_index=True,
            height=600,  # Set explicit height
            column_config={
                "Deal ID": st.column_config.TextColumn("Deal ID", width=120),
                "Customer": st.column_config.TextColumn("Customer", width=180),
                "Vehicle": st.column_config.TextColumn("Vehicle", width=150), 
                "Notes": st.column_config.TextColumn("Notes", width=350),
                "Summary": st.column_config.TextColumn("Summary", width=250),
                "Category": st.column_config.TextColumn("Category", width=150),
                "Next Steps": st.column_config.TextColumn("Next Steps", width=200)
            }
        )
        
        # Alternative: Display as expandable sections for better readability
        st.markdown("---")
        st.markdown("#### Detailed View (Expandable)")
        
        for i, result in enumerate(results):
            with st.expander(f"{result['Customer']} - {result['Vehicle']} - {result['Category']}"):
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.write(f"**Deal ID:** {result['Deal ID']}")
                    st.write(f"**Customer:** {result['Customer']}")
                    st.write(f"**Vehicle:** {result['Vehicle']}")
                    st.write(f"**Category:** {result['Category']}")
                with col2:
                    st.write(f"**Summary:** {result['Summary']}")
                    st.write(f"**Next Steps:** {result['Next Steps']}")
                st.write(f"**Full Notes:**")
                st.text_area("", value=result['Notes'].replace(' | ', '\n'), height=150, key=f"notes_{i}", disabled=True)
        
        # Category breakdown with weekly analysis and clickable categories
        if len(results) > 1:
            st.markdown("#### Category Breakdown by Week")
            
            # Prepare data for weekly breakdown
            results_df['TD Date'] = pd.to_datetime([r.get('TD Date', 'Unknown') for r in results], errors='coerce')
            results_df['Week Starting'] = results_df['TD Date'].dt.to_period('W-MON').dt.start_time.dt.date
            
            # Create weekly breakdown
            weekly_breakdown = results_df.groupby(['Week Starting', 'Category']).size().unstack(fill_value=0)
            weekly_breakdown['Total'] = weekly_breakdown.sum(axis=1)
            
            # Add total row
            total_row = weekly_breakdown.sum()
            total_row.name = 'Total'
            weekly_breakdown = pd.concat([weekly_breakdown, total_row.to_frame().T])
            
            st.dataframe(weekly_breakdown, use_container_width=True)
            
            # Clickable category buttons
            st.markdown("#### Click on a category to see details:")
            
            categories = results_df["Category"].value_counts()
            cols = st.columns(min(len(categories), 4))
            
            for i, (category, count) in enumerate(categories.items()):
                with cols[i % 4]:
                    if st.button(f"{category} ({count})", key=f"cat_{i}"):
                        st.session_state["selected_category"] = category
        
        # Display selected category details
        if "selected_category" in st.session_state:
            selected_cat = st.session_state["selected_category"]
            st.markdown(f"#### Details for: {selected_cat}")
            
            # Filter results for selected category
            cat_results = [r for r in results if r["Category"] == selected_cat]
            
            # Create detailed table
            detailed_data = []
            for result in cat_results:
                # Format the test drive date
                td_date = result.get("TD Date", "Unknown")
                if pd.notna(td_date) and td_date != "Unknown":
                    try:
                        if isinstance(td_date, str):
                            td_date = pd.to_datetime(td_date).strftime("%d %b %Y")
                        else:
                            td_date = td_date.strftime("%d %b %Y")
                    except:
                        td_date = str(td_date)
                
                detailed_data.append({
                    "Customer": result["Customer"],
                    "TD Date": td_date,
                    "Vehicles & IDs": result["Vehicle"],
                    "Notes Summary": result["Summary"],
                    "Next Steps": result["Next Steps"]
                })
            
            detailed_df = pd.DataFrame(detailed_data)
            st.dataframe(
                detailed_df, 
                use_container_width=True, 
                hide_index=True,
                column_config={
                    "Customer": st.column_config.TextColumn("Customer", width=180),
                    "TD Date": st.column_config.TextColumn("TD Date", width=120),
                    "Vehicles & IDs": st.column_config.TextColumn("Vehicles & IDs", width=280),
                    "Notes Summary": st.column_config.TextColumn("Notes Summary", width=350),
                    "Next Steps": st.column_config.TextColumn("Next Steps", width=180)
                }
            )
            
            if st.button("Clear Selection", key="clear_cat"):
                del st.session_state["selected_category"]
                st.rerun()

# ============ Rendering helpers ============
def header():
    cols = st.columns([1, 6, 1.2])
    with cols[0]:
        # Use absolute path to find H2.svg
        import os
        
        # Get the directory where app.py is located
        app_dir = os.path.dirname(os.path.abspath(__file__))
        logo_path = os.path.join(app_dir, "H2.svg")
        
        # Try to load the logo, fallback to text if it fails
        try:
            if os.path.exists(logo_path):
                st.image(logo_path, width=200, use_container_width=False)
            else:
                # Fallback to text logo
                st.markdown(
                    f"<div style='height:40px;display:flex;align-items:center;'><div style='background:{PRIMARY};padding:6px 10px;border-radius:6px;'><span style='font-weight:800;color:#FFFFFF'>CARS24</span></div></div>",
                    unsafe_allow_html=True
                )
        except Exception:
            # Fallback to text logo if any error occurs
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
        st.caption(f"🔄 Deployed: {DEPLOYMENT_TIME}")
    st.markdown('<hr class="div"/>', unsafe_allow_html=True)

def ctas():
    c1,c2 = st.columns(2)
    with c1:
        if st.button("🛣️  Test Drive Reminders\n\n• Friendly reminders  • TD date + state", key="cta1"):
            st.session_state["view"]="reminders"
        if st.button("👔  Manager Follow-Ups\n\n• After TD conducted  • Single date or range", key="cta2"):
            st.session_state["view"]="manager"
    with c2:
        if st.button("🕰️  Old Leads by Appointment ID\n\n• Re-engage older enquiries  • Skips active purchases", key="cta3"):
            st.session_state["view"]="old"
        if st.button("📊  Unsold TD Summary\n\n• ChatGPT analysis  • Date range + ticket owner", key="cta4"):
            st.session_state["view"]="unsold_summary"
    
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
    
    # Configure column widths based on content type
    column_config = {}
    for col in selected:
        if col in ["hs_object_id", "appointment_id"]:
            column_config[rename.get(col, col)] = st.column_config.TextColumn(rename.get(col, col), width=120)
        elif col in ["full_name", "email"]:
            column_config[rename.get(col, col)] = st.column_config.TextColumn(rename.get(col, col), width=200)
        elif col in ["vehicle_make", "vehicle_model"]:
            column_config[rename.get(col, col)] = st.column_config.TextColumn(rename.get(col, col), width=150)
        else:
            column_config[rename.get(col, col)] = st.column_config.TextColumn(rename.get(col, col), width=120)
    
    st.dataframe(
        disp[selected].rename(columns=rename), 
        use_container_width=True,
        column_config=column_config,
        height=400
    )

def render_selectable_messages(messages_df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Shows a data_editor with a checkbox per row; returns the edited DF.
       - Default = UNCHECKED
       - Forces text wrapping in the 'SMS draft' column for readability
    """
    if messages_df is None or messages_df.empty:
        st.info("No messages to preview."); return pd.DataFrame()

    view_df = messages_df[["CustomerName","Phone","Message"]].rename(
        columns={"CustomerName":"Customer","Message":"SMS draft"}
    ).copy()
    if "Send" not in view_df.columns:
        view_df.insert(0, "Send", False)  # default UNCHECKED

    edited = st.data_editor(
        view_df,
        key=f"editor_{key}",
        use_container_width=True,
        height=400,
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

        # 1) NEW: Filter out deals where SMS was already sent
        deals_not_sent, removed_sms_sent = filter_sms_already_sent(deals)

        # 2) Filter by appointment_id car active purchases
        deals_car_filtered, dropped_car_purchases = filter_deals_by_appointment_id_car_active_purchases(deals_not_sent)

        # 3) Filter internal/test emails
        deals_f, removed_internal = filter_internal_test_emails(deals_car_filtered)

        # 4) Audit dedupe
        dedup, dedupe_dropped = dedupe_users_with_audit(deals_f, use_conducted=False)

        # 5) Build messages with audit
        msgs, skipped_msgs = build_messages_with_audit(dedup, mode="reminder")

        # Store all artifacts
        st.session_state["reminders_deals"] = deals_f
        st.session_state["reminders_removed_sms_sent"] = removed_sms_sent  # NEW
        st.session_state["reminders_dropped_car_purchases"] = dropped_car_purchases
        st.session_state["reminders_removed_internal"] = removed_internal
        st.session_state["reminders_dedup"] = dedup
        st.session_state["reminders_dedupe_dropped"] = dedupe_dropped
        st.session_state["reminders_msgs"]  = msgs
        st.session_state["reminders_skipped_msgs"] = skipped_msgs
        
        # Store phone-to-deals mapping for later update
        st.session_state["reminders_phone_to_deals"] = get_all_deal_ids_for_contacts(msgs, deals_f)

    # ---- Render from persisted state ----
    deals_f      = st.session_state.get("reminders_deals")
    removed_sms  = st.session_state.get("reminders_removed_sms_sent")  # NEW
    dropped_car  = st.session_state.get("reminders_dropped_car_purchases")
    removed_int  = st.session_state.get("reminders_removed_internal")
    dedup        = st.session_state.get("reminders_dedup")
    dedupe_drop  = st.session_state.get("reminders_dedupe_dropped")
    msgs         = st.session_state.get("reminders_msgs")
    skipped_msgs = st.session_state.get("reminders_skipped_msgs")

    # NEW: Show SMS already sent filter results FIRST
    if isinstance(removed_sms, pd.DataFrame) and not removed_sms.empty:
        st.warning(f"⚠️ {len(removed_sms)} deals excluded - SMS reminders already sent")
        show_removed_table(removed_sms, "Removed (SMS reminder already sent)")

    # Show car purchase filter results
    if isinstance(dropped_car, pd.DataFrame) and not dropped_car.empty:
        show_removed_table(dropped_car, "Removed (car has active purchase deal via appointment_id)")

    if isinstance(removed_int, pd.DataFrame) and not removed_int.empty:
        show_removed_table(removed_int, "Removed by domain filter (cars24.com / yopmail.com)")

    if isinstance(deals_f, pd.DataFrame) and not deals_f.empty:
        # Update column display to include SMS status
        render_trimmed(deals_f, "Filtered deals (trimmed)", [
            ("hs_object_id","Deal ID"), 
            ("appointment_id","Appointment ID"), 
            ("full_name","Customer"), 
            ("email","Email"), 
            ("phone_norm","Phone"),
            ("vehicle_make","Make"), 
            ("vehicle_model","Model"),
            ("slot_date_prop","TD booking date"), 
            ("slot_time_param","Time"),
            ("video_url__short_","Video URL"),
            ("td_reminder_sms_sent","SMS Sent"),  # NEW
            ("Stage","Stage"),
        ])

    if isinstance(dedupe_drop, pd.DataFrame) and not dedupe_drop.empty:
        show_removed_table(dedupe_drop, "Collapsed during dedupe (duplicates)")

    if isinstance(dedup, pd.DataFrame) and not dedup.empty:
        st.markdown("#### <span style='color:#000000;'>Deduped list (by mobile|email)</span>", unsafe_allow_html=True)
        st.dataframe(dedup[["CustomerName","Phone","Email","DealsCount","Cars","WhenExact","DealStages","VideoURLs"]]
                     .rename(columns={"WhenExact":"When (exact)","DealStages":"Stage(s)"}),
                     use_container_width=True)

    if isinstance(msgs, pd.DataFrame) and not msgs.empty:
        st.markdown("#### <span style='color:#000000;'>Message Preview (Reminders)</span>", unsafe_allow_html=True)
        edited = render_selectable_messages(msgs, key="reminders")
        if isinstance(skipped_msgs, pd.DataFrame) and not skipped_msgs.empty:
            st.markdown("**Skipped while creating SMS**")
            st.dataframe(skipped_msgs, use_container_width=True)

        # MODIFIED: Send SMS button with deal update functionality
        if not edited.empty and st.button("Send SMS"):
            to_send = edited[edited["Send"]]
            if to_send.empty:
                st.warning("No rows selected.")
            elif not (AIRCALL_ID and AIRCALL_TOKEN and AIRCALL_NUMBER_ID):
                st.error("Missing Aircall credentials in .env.")
            else:
                st.info("Sending messages…")
                sent, failed = 0, 0
                sent_phones = []  # Track which phones were sent successfully
                
                for _, r in to_send.iterrows():
                    ok, msg = send_sms_via_aircall(r["Phone"], r["SMS draft"], AIRCALL_NUMBER_ID)
                    if ok: 
                        sent += 1
                        sent_phones.append(r["Phone"])
                        st.success(f"✅ Sent to {r['Phone']}")
                    else:  
                        failed += 1
                        st.error(f"❌ Failed for {r['Phone']}: {msg}")
                    time.sleep(1)
                
                # NEW: Update deals in HubSpot after successful sends
                if sent_phones and st.session_state.get("reminders_phone_to_deals"):
                    st.info("Updating HubSpot deals...")
                    phone_to_deals = st.session_state["reminders_phone_to_deals"]
                    
                    all_deal_ids = []
                    for phone in sent_phones:
                        if phone in phone_to_deals:
                            all_deal_ids.extend(phone_to_deals[phone])
                    
                    if all_deal_ids:
                        update_success, update_fail = update_deals_sms_sent(all_deal_ids)
                        if update_success > 0:
                            st.success(f"✅ Updated {update_success} deals with SMS sent status")
                        if update_fail > 0:
                            st.warning(f"⚠️ Failed to update {update_fail} deals")
                
                if sent: st.balloons()
                st.success(f"🎉 Done! SMS Sent: {sent} | Failed: {failed}")

def view_manager():
    st.subheader("👔  Manager Follow-Ups")

    # ===== Filters form =====
    with st.form("manager_form"):
        st.markdown('<div class="form-row">', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns([1.4, 1.6, 1.6, 1.2])

        with c1:
            mode = st.radio("Mode", ["Single date", "Date range"], horizontal=True, index=1)

        today = datetime.now(MEL_TZ).date()
        if mode == "Single date":
            with c2:
                d1 = st.date_input("Date", value=today)
                d2 = d1
            with c3:
                pass
        else:
            with c2:
                d1 = st.date_input("Start date", value=today - timedelta(days=7))
            with c3:
                d2 = st.date_input("End date", value=today)

        # Vehicle state filter (kept same as your existing pattern)
        state_options = hs_get_deal_property_options("car_location_at_time_of_sale")
        values = [o["value"] for o in state_options] if state_options else []
        labels = [o["label"] for o in state_options] if state_options else []
        def_val = "VIC" if "VIC" in values else (values[0] if values else "")
        with c4:
            if labels:
                chosen_label = st.selectbox(
                    "Vehicle state",
                    labels,
                    index=(values.index("VIC") if "VIC" in values else 0)
                )
                label_to_val = {o["label"]: o["value"] for o in state_options}
                rem_state_val = label_to_val.get(chosen_label, def_val)
            else:
                rem_state_val = def_val

        st.markdown("</div>", unsafe_allow_html=True)
        go = st.form_submit_button("Fetch deals", use_container_width=True)

    # ===== Data fetch and preprocessing =====
    if go:
        with st.spinner("Fetching deals from HubSpot…"):
            # Keep your existing search signature; this call mirrors your current usage.
            # If your function signature differs, keep your original argument list here.
            raw = hs_search_deals_by_date_property(
                date_property="td_conducted_date",
                date_from=d1,
                date_to=d2,
                pipeline_id=PIPELINE_ID,
                state_value=rem_state_val,
            )

        # Ensure expected columns and normalize (your existing helper)
        deals0 = prepare_deals(raw)

        # 1) Exclude contacts that have other ACTIVE purchase deals (kept from your existing logic)
        kept = deals0.copy()
        dropped_active = pd.DataFrame()
        try:
            deal_ids = [str(x) for x in kept["hs_object_id"].dropna().astype(str).tolist()]
            if deal_ids:
                d2c = hs_deals_to_contacts_map(deal_ids)
                contact_ids = sorted(set([c for v in d2c.values() for c in v]))
                c2d = hs_contacts_to_deals_map(contact_ids)
                other_deal_ids = sorted(set([did for v in c2d.values() for did in v if did not in deal_ids]))
                if other_deal_ids:
                    stage_map = hs_batch_read_deals(other_deal_ids, props=["dealstage"])
                    active_ids = set(
                        did for did, props in stage_map.items()
                        if (props or {}).get("dealstage") in ACTIVE_PURCHASE_STAGES
                    )
                    # mark any rows whose contact has an active purchase deal elsewhere
                    mask_drop = pd.Series(False, index=kept.index)  # start as all False
                    if d2c:
                        # any contact whose any other deal is active
                        contact_has_active = {
                            cid: any(d in active_ids for d in c2d.get(cid, []))
                            for cid in contact_ids
                        }
                        kept["__contact_has_active__"] = kept["hs_object_id"].map(
                            lambda did: any(contact_has_active.get(cid, False) for cid in d2c.get(str(did), []))
                        )
                        mask_drop = kept["__contact_has_active__"] == True
                    dropped_active = kept[mask_drop].copy()
                    if not dropped_active.empty:
                        dropped_active["Reason"] = "Active purchase on another deal"
                        kept = kept[~mask_drop].copy()
                        kept.drop(columns=["__contact_has_active__"], errors="ignore", inplace=True)
        except Exception as e:
            st.warning(f"Active purchase filter skipped due to: {e}")

        # ======== NEW: exclude deals where manager follow-up SMS already sent ========
        deals_not_sent, removed_sms_sent = filter_manager_sms_already_sent(kept)

        # 2) Internal/test email filter (your existing helper)
        deals_f, removed_internal = filter_internal_test_emails(deals_not_sent)

        # 3) Dedupe + message building (manager mode)
        dedup, dedupe_drop = dedupe_users_with_audit(deals_f, use_conducted=True)
        msgs, skipped_msgs = build_messages_with_audit(dedup, mode="manager")

        # Persist for downstream UI/actions
        st.session_state["manager_deals"] = deals_f
        st.session_state["manager_msgs"] = msgs
        st.session_state["manager_skipped_msgs"] = skipped_msgs

        # Store phone→deal mapping for post-send HubSpot updates
        st.session_state["manager_phone_to_deals"] = get_all_deal_ids_for_contacts(msgs, deals_f)

        # Persist removed buckets for “Reasons” UI
        st.session_state["manager_removed_sms_sent"] = removed_sms_sent
        st.session_state["manager_removed_internal"] = removed_internal
        st.session_state["manager_dropped_active"] = dropped_active
        st.session_state["manager_dedupe_drop"] = dedupe_drop

    # ===== Show removed buckets (Reasons) =====
    removed_sms = st.session_state.get("manager_removed_sms_sent")
    removed_int = st.session_state.get("manager_removed_internal")
    dropped_active = st.session_state.get("manager_dropped_active")
    dedupe_drop = st.session_state.get("manager_dedupe_drop")

    if isinstance(removed_sms, pd.DataFrame) and not removed_sms.empty:
        st.warning(f"⚠️ {len(removed_sms)} deals excluded - manager follow-up already sent")
        show_removed_table(removed_sms, "Removed (manager follow-up already sent)")

    if isinstance(dropped_active, pd.DataFrame) and not dropped_active.empty:
        show_removed_table(dropped_active, "Removed (active purchase on another deal)")

    if isinstance(removed_int, pd.DataFrame) and not removed_int.empty:
        show_removed_table(removed_int, "Removed by domain filter (cars24.com / yopmail.com)")

    if isinstance(dedupe_drop, pd.DataFrame) and not dedupe_drop.empty:
        show_removed_table(dedupe_drop, "Collapsed during dedupe (duplicates)")

    # ===== Deal preview =====
    deals_f = st.session_state.get("manager_deals")
    if isinstance(deals_f, pd.DataFrame) and not deals_f.empty:
        render_trimmed(
            deals_f,
            "Filtered deals (trimmed)",
            [
                ("hs_object_id", "Deal ID"),
                ("full_name", "Name"),
                ("email", "Email"),
                ("phone", "Phone"),
                ("vehicle_year", "Year"),
                ("vehicle_make", "Make"),
                ("vehicle_model", "Model"),
                ("car_location_at_time_of_sale", "State"),
                ("td_conducted_date", "TD conducted"),
                ("manager_followup_sms_sent", "Manager SMS sent"),
            ],
        )
        
    # ===== Messages table / send flow =====
    msgs = st.session_state.get("manager_msgs")
    skipped_msgs = st.session_state.get("manager_skipped_msgs")

    if isinstance(msgs, pd.DataFrame) and not msgs.empty:
        edited = render_selectable_messages(msgs, key="manager")

        if isinstance(skipped_msgs, pd.DataFrame) and not skipped_msgs.empty:
            st.dataframe(skipped_msgs, use_container_width=True)

        c1, c2, c3 = st.columns([1.2, 1, 1])
        with c1:
            send_btn = st.button("Send selected via Aircall", use_container_width=True)
        with c2:
            selected_only = st.checkbox("Only selected rows", value=True)
        with c3:
            st.caption("Uses Aircall number 2")

        if send_btn:
            to_send = edited[edited["Selected"] == True] if selected_only else edited.copy()
            if to_send.empty:
                st.warning("No rows selected.")
            else:
                st.info("Sending messages…")
                sent, failed = 0, 0
                sent_phones = []  # phones that succeeded

                for _, r in to_send.iterrows():
                    ok, msg = send_sms_via_aircall(r["Phone"], r["SMS draft"], AIRCALL_NUMBER_ID_2)
                    if ok:
                        sent += 1
                        sent_phones.append(r["Phone"])
                        st.success(f"✅ Sent to {r['Phone']}")
                    else:
                        failed += 1
                        st.error(f"❌ Failed for {r['Phone']}: {msg}")
                    time.sleep(1)

                # After successful sends, update HubSpot deals with manager_followup_sms_sent = true
                if sent_phones and st.session_state.get("manager_phone_to_deals"):
                    st.info("Updating HubSpot deals…")
                    phone_to_deals = st.session_state["manager_phone_to_deals"]

                    all_deal_ids = []
                    for phone in sent_phones:
                        if phone in phone_to_deals:
                            all_deal_ids.extend(phone_to_deals[phone])

                    if all_deal_ids:
                        update_success, update_fail = update_deals_manager_followup_sms_sent(all_deal_ids)
                        if update_success > 0:
                            st.success(f"✅ Updated {update_success} deals with manager follow-up SMS status")
                        if update_fail > 0:
                            st.warning(f"⚠️ Failed to update {update_fail} deals")

                if sent:
                    st.balloons()
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
            print(f"DEBUG: Checking {len(deal_ids)} deals: {deal_ids}")
            
            d2c = hs_deals_to_contacts_map(deal_ids)
            print(f"DEBUG: Deal-to-contact mapping: {d2c}")
            
            contact_ids = sorted({cid for cids in d2c.values() for cid in cids})
            print(f"DEBUG: Found {len(contact_ids)} contacts: {contact_ids}")
            
            c2d = hs_contacts_to_deals_map(contact_ids)
            print(f"DEBUG: Contact-to-deals mapping:")
            for cid, deal_list in c2d.items():
                print(f"  Contact {cid}: {deal_list}")
            
            other_deal_ids = sorted({did for _, dlist in c2d.items() for did in dlist if did not in deal_ids})
            print(f"DEBUG: Found {len(other_deal_ids)} other deals to check stages")
            
            stage_map = hs_batch_read_deals(other_deal_ids, props=["dealstage"])
            print(f"DEBUG: Retrieved stages for {len(stage_map)} deals")

            exclude_contacts = set()
            for cid, dlist in c2d.items():
                active_deals = []
                for did in dlist:
                    if did in deal_ids: continue
                    stage = (stage_map.get(did, {}) or {}).get("dealstage")
                    if stage and str(stage) in ACTIVE_PURCHASE_STAGE_IDS:
                        active_deals.append(f"{did}({stage})")
                        exclude_contacts.add(cid)
                        
                if active_deals:
                    print(f"DEBUG: Excluding contact {cid} - active purchases: {active_deals}")

            print(f"DEBUG: Total contacts to exclude: {len(exclude_contacts)}")

            def keep_row(row):
                d_id = str(row.get("hs_object_id") or "")
                cids = d2c.get(d_id, [])
                should_exclude = any((c in exclude_contacts) for c in cids)
                if should_exclude:
                    print(f"DEBUG: Excluding deal {d_id} - contact(s) {cids} have active purchases")
                return not should_exclude

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
            ("hs_object_id","Deal ID"), ("appointment_id","Appointment ID"), ("full_name","Customer"), ("email","Email"), ("phone_norm","Phone"),
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
                    ok, msg = send_sms_via_aircall(r["Phone"], r["SMS draft"], AIRCALL_NUMBER_ID_2)
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
    force_light_theme()  # Add this line
    v = st.session_state.get("view","home")
    if v == "home":
        ctas()
    elif v == "reminders":
        view_reminders()
    elif v == "manager":
        view_manager()
    elif v == "old":
        view_old()
    elif v == "unsold_summary":
        view_unsold_summary()

header_and_route()
