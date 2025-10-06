"""Streamlit entrypoint — routes to modular workflows (parity preserved)."""

import streamlit as st
from workflows.reminders import view_reminders
from workflows.manager import view_manager
from workflows.old_leads import view_old
from workflows.unsold_summary import view_unsold_summary

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

