from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun

model = ChatOllama(model="gemma3:4b")


def internet_search(query: str):
    """Run a web search
    Args:
        query (str): The query presented by user to search on internet.
    """
    search = DuckDuckGoSearchRun()
    return search.run(query)


def create_markdown_file(filename: str, content: str):
    """
    Creates a Markdown (.md) file with the given content.

    Args:
        filename (str): The name of the Markdown file (without .md extension is okay).
        content (str): The text to write into the Markdown file.        
    """
    # Ensure the filename ends with '.md'
    if not filename.endswith(".md"):
        filename += ".md"

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write(content)
        return f"✅ Markdown file '{filename}' created successfully!"
    except Exception as e:
        return f"❌ Error creating Markdown file: {e}"
