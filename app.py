import os
import sys
import argparse
from dotenv import load_dotenv
import colorama
from colorama import Fore, Style

# Initialize colorama for colored outputs
colorama.init(autoreset=True)

# Append directory to sys.path to ensure module resolution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logger
from tools.wikipedia_tool import get_wikipedia_tool
from tools.calculator_tool import get_calculator_tool
from tools.currency_tool import currency_converter
from tools.csv_tool import csv_country_lookup

from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq

def print_header(title: str):
    print(f"\n{Fore.CYAN}{Style.BRIGHT}{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}{Style.RESET_ALL}")

def print_success(message: str):
    print(f"{Fore.GREEN}{Style.BRIGHT}{message}{Style.RESET_ALL}")

def print_info(message: str):
    print(f"{Fore.BLUE}{message}{Style.RESET_ALL}")

def print_warning(message: str):
    print(f"{Fore.YELLOW}{Style.BRIGHT}{message}{Style.RESET_ALL}")

def print_error(message: str):
    print(f"{Fore.RED}{Style.BRIGHT}{message}{Style.RESET_ALL}")

def run_predefined_tests(agent):
    test_queries = [
        "What is the GDP of Japan in EUR?",
        "Who founded Microsoft and how old would he be 50 years after Microsoft was founded?",
        "What is the capital of Germany and what is 20% of its GDP?",
        "Convert Egypt's GDP from USD to EGP",
        "If Japan's GDP grows by 5%, what will it become?"
    ]
    
    print_header("RUNNING PREDEFINED TEST SUITE")
    print_info(f"Writing all reasoning traces to: traces/reasoning_logs.txt\n")
    
    for idx, query in enumerate(test_queries, 1):
        print(f"\n{Fore.MAGENTA}{Style.BRIGHT}[Test {idx}/5] Query: {query}{Style.RESET_ALL}")
        print(f"{Fore.LIGHTBLACK_EX}{'-'*80}{Style.RESET_ALL}")
        try:
            # We call the agent. The stdout verbose=True logs are intercepted by DualLogger
            response = agent.run(query)
            print_success(f"\n[Result {idx}]: {response}")
        except Exception as e:
            print_error(f"\nError running query '{query}': {str(e)}")
        print(f"{Fore.LIGHTBLACK_EX}{'-'*80}{Style.RESET_ALL}\n")

def interactive_loop(agent):
    print_header("INTERACTIVE RESEARCH ASSISTANT CLI")
    print_info("Type your research questions below. Enter 'exit', 'quit', or 'q' to end the session.\n")
    
    while True:
        try:
            user_query = input(f"{Fore.GREEN}{Style.BRIGHT}Research Question > {Style.RESET_ALL}").strip()
            if not user_query:
                continue
            if user_query.lower() in ["exit", "quit", "q"]:
                print_info("Exiting interactive session. Goodbye!")
                break
                
            print(f"\n{Fore.LIGHTBLACK_EX}{'-'*80}{Style.RESET_ALL}")
            response = agent.run(user_query)
            print_success(f"\nAnswer: {response}")
            print(f"{Fore.LIGHTBLACK_EX}{'-'*80}{Style.RESET_ALL}\n")
            
        except KeyboardInterrupt:
            print_info("\nExiting interactive session. Goodbye!")
            break
        except Exception as e:
            print_error(f"An error occurred: {str(e)}\n")

def main():
    # Setup command-line argument parsing
    parser = argparse.ArgumentParser(description="Multi-Tool Research Assistant using LangChain & Groq")
    parser.add_argument("--run-tests", action="store_true", help="Run the predefined 5 test queries and exit immediately.")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()
    
    # Initialize Dual Logger to redirect console output to file
    setup_logger("traces/reasoning_logs.txt")
    
    print_header("INITIALIZING MULTI-TOOL RESEARCH ASSISTANT")
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_api_key_here":
        print_error("Error: GROQ_API_KEY is not set in the environment or .env file.")
        print_info("Please set the GROQ_API_KEY variable inside multi-tool-research-assistant/.env file.")
        sys.exit(1)
        
    model_name = os.getenv("GROQ_MODEL", "llama3-70b-8192")
    
    print_info(f"LLM Model: {model_name}")
    print_info("Loading tools...")
    
    try:
        # Initialize Groq LLM
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name=model_name,
            temperature=0.0
        )
        
        # Initialize tools
        tools = [
            get_wikipedia_tool(),
            get_calculator_tool(llm),
            currency_converter,
            csv_country_lookup
        ]
        
        print_success("Tools loaded successfully:")
        for t in tools:
            print(f" - {t.name}: {t.description[:80]}...")
            
        # Initialize ReAct Agent
        print_info("Initializing ReAct agent...")
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
        print_success("Agent initialized successfully.")
        
    except Exception as e:
        print_error(f"Failed to initialize Agent: {str(e)}")
        sys.exit(1)
        
    if args.run_tests:
        run_predefined_tests(agent)
        return
        
    # Main menu loop if not running test mode directly
    while True:
        print_header("MAIN MENU")
        print("1. Run 5 Predefined Test Questions")
        print("2. Enter Interactive CLI Session")
        print("3. Exit")
        
        choice = input(f"\n{Fore.GREEN}{Style.BRIGHT}Select an option (1-3) > {Style.RESET_ALL}").strip()
        
        if choice == "1":
            run_predefined_tests(agent)
        elif choice == "2":
            interactive_loop(agent)
            break
        elif choice == "3":
            print_info("Goodbye!")
            break
        else:
            print_warning("Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()
