"""Workflow view: Manager Follow-ups — extracted from original app.py."""


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
            ("hs_object_id","Deal ID"), ("appointment_id","Appointment ID"), ("full_name","Customer"), ("email","Email"), ("phone_norm","Phone"),
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
                    ok, msg = send_sms_via_aircall(r["Phone"], r["SMS draft"], AIRCALL_NUMBER_ID_2)
                    if ok: sent += 1; st.success(f"✅ Sent to {r['Phone']}")
                    else:  failed += 1; st.error(f"❌ Failed for {r['Phone']}: {msg}")
                    time.sleep(1)
                if sent: st.balloons()
                st.success(f"🎉 Done! Sent: {sent} | Failed: {failed}")

