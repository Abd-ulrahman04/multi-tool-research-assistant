import os
import sys
import io
import contextlib
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tools.wikipedia_tool import get_wikipedia_tool
from tools.calculator_tool import get_calculator_tool
from tools.currency_tool import currency_converter
from tools.csv_tool import csv_country_lookup

from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _schedule_query(query: str) -> None:
    """Preset buttons: set query and flag execution (runs before widget sync)."""
    st.session_state.user_query = query
    st.session_state.execute_query = True


def _schedule_execute() -> None:
    """Execute button: run whatever is currently in the query box."""
    st.session_state.execute_query = True


def _run_agent(query: str, api_key: str, model_name: str) -> tuple[str, str]:
    """Run the ReAct agent and return (answer, reasoning trace)."""
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
        temperature=0.0,
    )
    tools = [
        get_wikipedia_tool(),
        get_calculator_tool(llm),
        currency_converter,
        csv_country_lookup,
    ]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    stdout_capture = io.StringIO()
    with contextlib.redirect_stdout(stdout_capture):
        response = agent.run(query)
    return response, stdout_capture.getvalue()


def _append_trace_log(query: str, trace_output: str) -> None:
    log_dir = os.path.join(CURRENT_DIR, "traces")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, "reasoning_logs.txt")
    with open(log_file_path, "a", encoding="utf-8") as f:
        f.write(f"\n[STREAMLIT RUN] Query: {query}\n")
        f.write("-" * 80 + "\n")
        f.write(trace_output)
        f.write("-" * 80 + "\n")


# --- Page Configuration ---
st.set_page_config(
    page_title="Multi-Tool Research Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Premium Custom Styling (Dark Mode / Glassmorphism) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono&display=swap');
    
    /* Global style overrides */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
        background-color: #0b0f19;
        color: #f1f5f9;
    }
    
    /* Sidebar customization */
    [data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #1e293b;
    }
    
    /* Custom Title Gradient */
    .title-text {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #a78bfa 0%, #3b82f6 50%, #10b981 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
        filter: drop-shadow(0 2px 8px rgba(59, 130, 246, 0.2));
    }
    
    .subtitle-text {
        font-size: 1.2rem;
        color: #94a3b8;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    /* Custom Card Style for Final Answers */
    .answer-card {
        background: rgba(30, 41, 59, 0.4);
        border: 1px solid rgba(167, 139, 250, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        margin-bottom: 15px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 0 10px rgba(167, 139, 250, 0.05);
    }
    
    /* Reasoning Trace Container */
    .trace-container {
        font-family: 'JetBrains Mono', monospace;
        background-color: #020617 !important;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 15px;
        color: #38bdf8;
        font-size: 0.9rem;
        line-height: 1.5;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Status indicators */
    .status-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #1e293b;
        border: 1px solid #334155;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    .status-active {
        color: #10b981;
        border-color: #065f46;
        background-color: rgba(16, 185, 129, 0.08);
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        color: #cbd5e1;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        text-align: left;
        padding: 10px 15px;
    }
    
    div.stButton > button:hover {
        border-color: #3b82f6;
        color: #f8fafc;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }
    
    /* Input customization */
    div.stTextInput input {
        background-color: #0f172a;
        border: 1px solid #334155;
        color: #f8fafc;
        border-radius: 8px;
    }
    
    div.stTextInput input:focus {
        border-color: #a78bfa;
        box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.25);
    }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar Configuration ---
with st.sidebar:
    st.markdown("<h3 style='color: #a78bfa;'>⚙️ Settings & Configuration</h3>", unsafe_allow_html=True)
    
    # API Key Handling
    groq_api_key_env = os.getenv("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=groq_api_key_env if groq_api_key_env else "your_api_key_here",
        type="password",
        help="Provide your Groq API Key. Pre-populated from local environmental settings if available."
    )
    
    # Model Selection
    model_options = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768"
    ]
    selected_model = st.selectbox(
        "Groq Model",
        options=model_options,
        index=0,
        help="Select the Groq model for the ReAct Agent's reasoning backbone."
    )
    
    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #3b82f6;'>🛡️ Active Agent Tools</h4>", unsafe_allow_html=True)
    st.markdown("""
        <span class='status-badge status-active'>📚 Wikipedia Lookup</span>
        <span class='status-badge status-active'>🧮 LLMMath Calculator</span>
        <span class='status-badge status-active'>💱 Currency Converter</span>
        <span class='status-badge status-active'>📊 CSV Database (pandas)</span>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: #334155;'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #10b981;'>📂 Local GDP Database</h4>", unsafe_allow_html=True)
    
    # Inspect data/countries.csv
    try:
        csv_path = os.path.join(CURRENT_DIR, "data", "countries.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            st.dataframe(
                df,
                column_config={
                    "Country": "Country Name",
                    "Capital": "Capital City",
                    "GDP": st.column_config.NumberColumn("GDP (USD)", format="$%,.0f")
                },
                hide_index=True
            )
        else:
            st.warning("countries.csv database not found.")
    except Exception as e:
        st.error(f"Error loading database: {str(e)}")

# --- Main Layout ---
st.markdown("<div class='title-text'>Multi-Tool Research Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>A LangChain ReAct AI Agent executing multi-step queries with live reasoning traces</div>", unsafe_allow_html=True)

# Initialize session state variables
if "user_query" not in st.session_state:
    st.session_state.user_query = ""
if "answer" not in st.session_state:
    st.session_state.answer = None
if "trace" not in st.session_state:
    st.session_state.trace = None
if "execute_query" not in st.session_state:
    st.session_state.execute_query = False
if "last_query" not in st.session_state:
    st.session_state.last_query = ""

PRESET_QUERIES = [
    ("🇯🇵 What is the GDP of Japan in EUR?", "What is the GDP of Japan in EUR?"),
    (
        "💻 Who founded Microsoft & how old is he 50 years later?",
        "Who founded Microsoft and how old would he be 50 years after Microsoft was founded?",
    ),
    (
        "🇩🇪 What is Germany's capital & 20% of its GDP?",
        "What is the capital of Germany and what is 20% of its GDP?",
    ),
    ("🇪🇬 Convert Egypt's GDP from USD to EGP", "Convert Egypt's GDP from USD to EGP"),
    ("📈 If Japan's GDP grows by 5%, what will it become?", "If Japan's GDP grows by 5%, what will it become?"),
]

# Predefined Questions Section
st.markdown("<h3 style='color: #cbd5e1; font-size: 1.3rem; font-weight: 600;'>💡 Preset Research Questions</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    for label, query in PRESET_QUERIES[:3]:
        st.button(label, on_click=_schedule_query, args=(query,), use_container_width=True)

with col2:
    for label, query in PRESET_QUERIES[3:]:
        st.button(label, on_click=_schedule_query, args=(query,), use_container_width=True)

st.markdown("<hr style='border-color: #1e293b;'>", unsafe_allow_html=True)

# Custom question: form avoids reruns on every keystroke; Enter submits
with st.form("research_query_form", clear_on_submit=False):
    st.text_input(
        "Ask the Research Assistant:",
        placeholder="E.g., What is the capital of France and what would its GDP be in GBP if converted?",
        key="user_query",
    )
    submitted = st.form_submit_button("⚡ Execute Research Query", use_container_width=True)

if submitted:
    _schedule_execute()

# Execute Query Block (triggered by preset or submit via session-state flag)
if st.session_state.execute_query:
    st.session_state.execute_query = False
    query_to_run = st.session_state.user_query.strip()

    if not query_to_run:
        st.warning("Please enter a research question before executing.")
    elif not api_key_input or api_key_input == "your_api_key_here":
        st.error("Please set a valid GROQ_API_KEY in the sidebar before running.")
    else:
        st.session_state.last_query = query_to_run
        st.session_state.answer = None
        st.session_state.trace = None

        with st.spinner("🧠 Analyzing and orchestrating reasoning plan..."):
            try:
                response, trace_output = _run_agent(
                    query_to_run, api_key_input, selected_model
                )
                st.session_state.answer = response
                st.session_state.trace = trace_output
                _append_trace_log(query_to_run, trace_output)
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# --- Always display results after a question has been answered ---
if st.session_state.answer is not None:
    if st.session_state.last_query:
        st.caption(f"Query: {st.session_state.last_query}")

    st.markdown("<h3 style='color: #10b981;'>🎯 Final Answer</h3>", unsafe_allow_html=True)
    st.markdown(
        "<div class='answer-card'><div style='font-size: 1.15rem; line-height: 1.6; color: #f8fafc;'>",
        unsafe_allow_html=True,
    )
    st.write(st.session_state.answer)
    st.markdown("</div></div>", unsafe_allow_html=True)

    if st.session_state.trace:
        st.markdown("<h3 style='color: #3b82f6;'>🔍 Agent Reasoning Trace</h3>", unsafe_allow_html=True)
        with st.expander("Show step-by-step thinking process", expanded=True):
            cleaned_trace = "\n".join(
                line for line in st.session_state.trace.splitlines() if line.strip()
            )
            st.code(cleaned_trace, language=None)

