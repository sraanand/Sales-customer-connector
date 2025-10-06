"""OpenAI-assisted drafting & message assembly — preserved."""
from config import *
import streamlit as st
import pandas as pd
try:
    import openai
except Exception:
    openai=None


# ---- _call_openai ----

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



# ---- _call_openai ----

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



# ---- draft_sms_reminder ----

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



# ---- draft_sms_manager ----

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



# ---- draft_sms_oldlead_by_stage ----

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



# ---- draft_sms_oldlead_by_stage_improved ----

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



# ---- build_messages_with_audit ----

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



# ---- build_messages_from_dedup ----

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


