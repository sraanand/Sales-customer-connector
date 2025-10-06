"""
config.py — Central configuration & constants
(extracted verbatim from original app.py where applicable)
"""

import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

HUBSPOT_TOKEN     = os.getenv("HUBSPOT_TOKEN", "")
AIRCALL_ID        = os.getenv("AIRCALL_ID")
AIRCALL_TOKEN     = os.getenv("AIRCALL_TOKEN")
AIRCALL_NUMBER_ID = os.getenv("AIRCALL_NUMBER_ID")
AIRCALL_NUMBER_ID_2 = os.getenv("AIRCALL_NUMBER_ID_2")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
DEPLOYMENT_TIME = datetime.now(ZoneInfo("Australia/Melbourne")).strftime("%Y-%m-%d %H:%M:%S AEST")
PREFERRED_MODELS = ["gpt-4o-mini", "o4-mini", "gpt-4o", "gpt-3.5-turbo"]
MEL_TZ = ZoneInfo("Australia/Melbourne")
UTC_TZ = timezone.utc
HS_ROOT       = "https://api.hubspot.com"
HS_SEARCH_URL = f"{HS_ROOT}/crm/v3/objects/deals/search"
HS_PROP_URL   = f"{HS_ROOT}/crm/v3/properties/deals"
HS_PAGE_LIMIT = 100
HS_TOTAL_CAP  = 1000
AIRCALL_BASE_URL = "https://api.aircall.io/v1"
PIPELINE_ID        = "2345821"
STAGE_ENQUIRY_ID   = "1119198251"
STAGE_BOOKED_ID    = "1119198252"
STAGE_CONDUCTED_ID = "1119198253"
OLD_LEAD_START_STAGES = {STAGE_ENQUIRY_ID, STAGE_BOOKED_ID, STAGE_CONDUCTED_ID}
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
]
STAGE_LABELS = {
    STAGE_ENQUIRY_ID:   "Enquiry (no TD)",
    STAGE_BOOKED_ID:    "TD booked",
    STAGE_CONDUCTED_ID: "TD conducted (no deposit)",
}
PRIMARY = "#4736FE"
