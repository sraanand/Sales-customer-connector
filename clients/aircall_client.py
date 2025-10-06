"""Aircall SMS send wrapper — copied 1:1 from original app.py."""
from config import *
import requests


# ---- send_sms_via_aircall ----

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


