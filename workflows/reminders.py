"""Workflow view: Test Drive Reminders — extracted from original app.py."""


from config import *
from clients.hubspot_client import *
from clients.aircall_client import *
from core.utils import *
from core.drafting import *
from ui.components import *
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo


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

