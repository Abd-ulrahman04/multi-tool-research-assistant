import wikipedia
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

def get_wikipedia_tool() -> WikipediaQueryRun:
    """
    Returns an instance of WikipediaQueryRun configured with a WikipediaAPIWrapper.
    """
    # Set a descriptive User-Agent as required by Wikimedia's API policy
    wikipedia.set_user_agent("MultiToolResearchAssistant/1.0 (https://github.com/example/multi-tool-research-assistant; research@example.com)")
    
    # top_k_results=2 helps get concise but comprehensive search answers.
    # doc_content_chars_max=3000 keeps context size reasonable.
    api_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=3000)
    return WikipediaQueryRun(
        name="Wikipedia",
        api_wrapper=api_wrapper,
        description="A wrapper around Wikipedia. Useful for searching general information, history, founders, biography, dates, and entities."
    )
