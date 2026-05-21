from langchain.agents import Tool
from langchain.chains import LLMMathChain
from langchain_core.language_models import BaseLanguageModel

def get_calculator_tool(llm: BaseLanguageModel) -> Tool:
    """
    Creates and returns a calculator Tool wrapped around LLMMathChain.
    """
    # LLMMathChain uses the LLM to write python code or solve math expressions
    llm_math = LLMMathChain.from_llm(llm=llm, verbose=True)
    
    return Tool(
        name="Calculator",
        func=llm_math.run,
        description=(
            "Useful for when you need to perform math calculations or evaluate mathematical expressions. "
            "Input should be a mathematical expression (e.g. '4200000000000 * 1.05' or '1975 + 50')."
        )
    )
