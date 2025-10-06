"""Workflow view: Old Leads by Appointment ID — extracted from original app.py."""


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

