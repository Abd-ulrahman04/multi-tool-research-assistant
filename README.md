# Multi-Tool Research Assistant

A complete, production-grade ReAct (Reasoning and Acting) AI Agent powered by LangChain and Groq API (using the `llama3-70b-8192` model). The agent is equipped with a suite of custom tools and executes complex, multi-step reasoning to answer factual, quantitative, and mathematical questions.

## Features

1. **ReAct Reasoning Agent**: Implements the ReAct paradigm using `initialize_agent` with verbose tracing enabled to make the agent's thought process completely transparent.
2. **Wikipedia Integration**: Searches and synthesizes summary facts from Wikipedia for queries about founders, dates, companies, and locations.
3. **LLMMath Calculator**: Integrates LangChain's `LLMMathChain` to perform precise arithmetic calculations (e.g., computing percentages, compound growth, ages).
4. **Exchange Rate Tool**: Calls the live `exchangerate-api` to convert monetary amounts between global currencies (e.g., converting USD to EUR or EGP).
5. **CSV Database Lookup**: Reads `data/countries.csv` using pandas to query countries, capitals, and GDP statistics.
6. **Dual Streaming Logger**: Intercepts standard output and redirects a clean version of all reasoning traces (stripping ANSI terminal colors) to `traces/reasoning_logs.txt` while keeping the terminal UI colorful and active.
7. **Interactive CLI Menu & Test Suite**: Run the built-in 5 test queries automatically, or use the interactive command-line interface to input custom queries.

---

## Folder Structure

```text
multi-tool-research-assistant/
│
├── app.py                   # Main program entrypoint (CLI & test suite runner)
├── .env.example             # Environment template (copy to .env)
├── requirements.txt         # Python dependency manifest
├── README.md                # Project documentation
│
├── data/
│   └── countries.csv        # Local country capital & GDP database
│
├── traces/
│   └── reasoning_logs.txt   # Output file containing reasoning trace logs
│
├── tools/
│   ├── wikipedia_tool.py    # Wikipedia Query Tool
│   ├── calculator_tool.py   # LLMMathChain Math Calculator Tool
│   ├── currency_tool.py     # Custom Live Currency Converter Tool
│   └── csv_tool.py          # Custom CSV pandas lookup tool
│
└── utils/
    └── logger.py            # Console/File redirection logging utility
```

---

## Technical Stack

* **Python** (version 3.10+)
* **LangChain** (agent orchestration)
* **langchain-groq** (Groq Llama-3 model client)
* **pandas** (CSV reading and querying)
* **requests** (live exchange rate API requests)
* **colorama** (colored terminal output)
* **python-dotenv** (loading configurations)

---

## Installation & Setup

1. **Navigate to the project directory**:
   ```bash
   cd multi-tool-research-assistant
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and set your Groq API key:
   ```bash
   cp .env.example .env
   ```
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=llama3-70b-8192
   ```

---

## How to Run

### Run the Streamlit Web Application (Recommended GUI)
To launch the premium glassmorphic Streamlit web application dashboard:
```bash
python -m streamlit run streamlit_app.py --server.fileWatcherType none
```
This runs the app locally at `http://localhost:8501`. It includes:
- Live database viewer showing the contents of `countries.csv`.
- Quick-click preset buttons for the 5 sample research questions.
- A clean interactive search query box.
- Dynamic expanding reasoning traces (real-time ReAct logs).
- API configurations directly customizable from the sidebar.

### Run the Interactive CLI Menu
Run the terminal-based choice menu:
```bash
python app.py
```
From here you can select:
* `1` to run the 5 predefined multi-step test queries.
* `2` to run the interactive CLI loop.
* `3` to exit.

### Headless/Automated Test Suite
To run the predefined tests automatically and print/log the traces directly to the console/file:
```bash
python app.py --run-tests
```

---

## How the ReAct Paradigm Works

The **ReAct (Reasoning + Acting)** framework enables the LLM to solve complex queries by alternating between reasoning steps ("Thoughts") and actions ("Actions"). 

1. **Thought**: The agent analyzes the question and makes a plan.
2. **Action**: The agent selects a Tool (e.g. Wikipedia, Calculator, Currency Converter, CSV Country Lookup) and provides arguments.
3. **Observation**: The tool executes and returns the result (the "Observation") to the agent.
4. **Repeat**: The agent inspects the observation, reasons on the next step, executes another action if needed, or generates the final response.

---

## Example Output & Traces

### Example: "What is the GDP of Japan in EUR?"
```text
> Entering new AgentExecutor chain...
Thought: I need to find the GDP of Japan first. I have a tool called csv_country_lookup which looks up country information from the local database. Let me query Japan.
Action: csv_country_lookup
Action Input: Japan
Observation: Country: Japan | Capital: Tokyo | GDP: $4,200,000,000,000.00 USD
Thought: Now I have the GDP of Japan, which is $4,200,000,000,000.00 USD. The query asks for the GDP in EUR. I have a tool called currency_converter which converts currency amounts. I should use this to convert $4.2e12 USD to EUR.
Action: currency_converter
Action Input: 4200000000000 USD to EUR
Observation: 4,200,000,000,000.00 USD is equal to 3,864,000,000,000.00 EUR (1 USD = 0.9200 EUR)
Thought: I have successfully converted the GDP of Japan from USD to EUR. The result is 3,864,000,000,000.00 EUR. I can now provide the final answer.
Final Answer: The GDP of Japan is approximately 3.86 trillion EUR (or exactly 3,864,000,000,000.00 EUR).
> Finished chain.
```

---

## Screenshots Placeholder
Here are placeholders for future screenshots representing terminal outputs and file logs:

* **CLI Main Menu**: `[Insert Main Menu screenshot here]`
* **ReAct Reasoning Chain**: `[Insert Verbose Reasoning Trace screenshot here]`
* **Interactive CLI loop**: `[Insert CLI prompt screenshot here]`
