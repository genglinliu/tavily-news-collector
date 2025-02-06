"""
starter code for tavily api
"""
import os
from tavily import TavilyClient


# Initialize the Tavily client with your API key
tavily_client = TavilyClient(api_key="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")
response = tavily_client.search("Who is Leo Messi?")

print(response)