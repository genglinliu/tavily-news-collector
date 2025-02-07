import pandas as pd
import os
from typing import Dict, Any
from tavily import TavilyClient
import json
from tqdm import tqdm
import random
from domains import fake_news_domains_newsguard, fake_news_domains_wiki

FAKE_NEWS_DOMAINS = random.sample(fake_news_domains_newsguard + fake_news_domains_wiki, 200)

class NewsSearch:

    def __init__(self, api_key: str = None):
        """Initialize with Tavily API key from environment variable or parameter."""
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Please provide TAVILY_API_KEY as environment variable or parameter")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str, include_domains: list[str] = None) -> Dict[str, Any]:
        """
        Perform a search prioritizing specific domains.
        
        Args:
            query: The search query string
            
        Returns:
            Dict containing search results
        """
        return self.client.search(
            query=query,
            include_domains=include_domains,
            max_results=20
        )

def print_results(results: Dict[str, Any]) -> None:
    """Print search results in a clean format."""
    print("\nSearch Results:")
    print("-" * 80)
    
    for result in results['results']:
        print(f"\nTitle: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Content: {result['content']}")
        print("-" * 80)

def enrich_claim_with_sources(searcher: NewsSearch, claim: str) -> Dict[str, Any]:
    """
    Enrich a claim with supporting and opposing sources by performing searches.
    
    Args:
        searcher: NewsSearch instance
        claim: The claim to search for
        
    Returns:
        Dict containing supporting_sources and opposing_sources
    """
    # Search reliable sources
    reliable_results = searcher.search(claim, include_domains=None)
    
    # Search potentially misleading sources
    misinfo_results = searcher.search(claim, include_domains=FAKE_NEWS_DOMAINS)
    
    # Format results into supporting and opposing sources
    supporting_sources = [
        {
            "title": result["title"],
            "url": result["url"],
            "content": result["content"]
        }
        for result in reliable_results["results"]  
    ]
    
    opposing_sources = [
        {
            "title": result["title"],
            "url": result["url"],
            "content": result["content"]
        }
        for result in misinfo_results["results"]  
    ]
    
    return {
        "supporting_sources": supporting_sources,
        "opposing_sources": opposing_sources
    }

def enrich_covid_data(input_file: str, output_file: str, api_key: str = None):
    """
    Read COVID data, enrich with sources, and append to output file as we go.
    
    Args:
        input_file: Path to input JSON file
        output_file: Path to output JSON file
        api_key: Optional Tavily API key
    """
    # Initialize search
    searcher = NewsSearch(api_key=api_key)
    
    # Read input data
    with open(input_file, 'r') as f:
        covid_data = json.load(f)
    
    # Create/clear output file with empty array
    with open(output_file, 'w') as f:
        f.write('[\n')
    
    # Process each claim and append to file
    for i, item in enumerate(tqdm(covid_data)):
        try:
            sources = enrich_claim_with_sources(searcher, item["claim"])
            item.update(sources)
            
            # Append to file with proper JSON formatting
            with open(output_file, 'a') as f:
                if i > 0:
                    f.write(',\n')
                json.dump(item, f, indent=4)
                
        except Exception as e:
            print(f"\nError processing claim: {item['claim']}")
            print(f"Error: {str(e)}")
            raise e
    
    # Close the JSON array
    with open(output_file, 'a') as f:
        f.write('\n]')
    
    print("\nCompleted processing all items")

if __name__ == "__main__":
    # Example usage
    input_file = "data/final-data/final_climate_data.json"
    output_file = "data/final-data/enriched_climate_data.json"
    api_key = "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"
    
    enrich_covid_data(input_file, output_file, api_key)
