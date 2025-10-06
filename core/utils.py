"""General utilities (dates, filters, dedupe) — logic preserved."""
from config import *
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
import streamlit as st


# ---- mel_day_bounds_to_epoch_ms ----

def mel_day_bounds_to_epoch_ms(d: date) -> tuple[int, int]:
    start_local = datetime(d.year, d.month, d.day, 0, 0, 0, tzinfo=MEL_TZ)
    end_local   = start_local + timedelta(days=1) - timedelta(milliseconds=1)
    start_ms    = int(start_local.astimezone(UTC_TZ).timestamp() * 1000)
    end_ms      = int(end_local.astimezone(UTC_TZ).timestamp() * 1000)
    return start_ms, end_ms



# ---- parse_epoch_or_iso_to_local_date ----

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



# ---- normalize_phone ----

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



# ---- format_date_au ----

def format_date_au(d: date) -> str:
    return d.strftime("%d %b %Y") if isinstance(d, date) else ""



# ---- rel_date ----

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



# ---- prepare_deals ----

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



# ---- filter_internal_test_emails ----

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



# ---- filter_sms_already_sent ----

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



# ---- get_all_deal_ids_for_contacts ----

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



# ---- dedupe_users_with_audit ----

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



# ---- dedupe_users ----

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


