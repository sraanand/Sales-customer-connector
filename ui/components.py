"""Streamlit table renderers & selectors — preserved."""
import streamlit as st
import pandas as pd


# ---- show_removed_table ----

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



# ---- render_trimmed ----

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



# ---- render_selectable_messages ----

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


