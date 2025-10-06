"""Workflow view: Unsold TD Summary — extracted from original app.py."""


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

