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
            include_domains: Optional list of domains to prioritize
            
        Returns:
            Dict containing search results with detailed content
        """
        return self.client.search(
            query=query,
            include_domains=include_domains,
            max_results=20,
            search_depth="advanced",  # Get detailed content and analysis
            include_raw_content=True,
        )


def enrich_claim_with_sources(searcher: NewsSearch, claim: str, max_sources: int = 40) -> Dict[str, Any]:
    """
    Enrich a claim with supporting and opposing sources by performing searches.
    
    Args:
        searcher: NewsSearch instance
        claim: The claim to search for
        max_sources: Maximum number of sources to collect for each category (default: 40)
        
    Returns:
        Dict containing supporting_sources and opposing_sources (up to max_sources each)
    """
    # Collect supporting sources through multiple searches
    supporting_sources = []
    seen_supporting_urls = set()
    
    # Try a few different search variations to get more diverse results
    search_variations = [
        claim,
        f"fact check {claim}",
        f"evidence {claim}",
        f"research {claim}"
    ]
    
    for query in search_variations:
        if len(supporting_sources) >= max_sources:
            break
            
        results = searcher.search(query, include_domains=None)
        
        # Add new unique results
        for result in results["results"]:
            if result["url"] not in seen_supporting_urls and len(supporting_sources) < max_sources:
                supporting_sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "content": result["content"]
                })
                seen_supporting_urls.add(result["url"])
                print(f"Added supporting source: {result['url']}")
    print(f"Found {len(supporting_sources)} supporting sources")    
    
    # Collect opposing sources through multiple searches
    opposing_sources = []
    seen_opposing_urls = set()
    domains_to_try = FAKE_NEWS_DOMAINS.copy()
    
    while len(opposing_sources) < max_sources and domains_to_try:
        # Take next batch of domains to try
        current_domains = domains_to_try[:200]
        domains_to_try = domains_to_try[200:]
        
        # Search potentially misleading sources
        misinfo_results = searcher.search(claim, include_domains=current_domains)
        
        # Add new unique results
        for result in misinfo_results["results"]:
            if result["url"] not in seen_opposing_urls and len(opposing_sources) < max_sources:
                opposing_sources.append({
                    "title": result["title"],
                    "url": result["url"],
                    "content": result["content"]
                })
                seen_opposing_urls.add(result["url"])
                print(f"Added opposing source: {result['url']}")
            
        print(f"Found {len(misinfo_results['results'])} opposing sources, trying again")
                
    print(f"Found {len(opposing_sources)} opposing sources")
    
    return {
        "supporting_sources": supporting_sources,
        "opposing_sources": opposing_sources
    }

def enrich_data(input_file: str, output_file: str, api_key: str = None, max_sources: int = 40):
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
        data = json.load(f)
        
    data = data[:50]
    
    # Create/clear output file with empty array
    with open(output_file, 'w') as f:
        f.write('[\n')
    
    # Process each claim and append to file
    for i, item in enumerate(tqdm(data)):
        try:
            sources = enrich_claim_with_sources(searcher, item["claim"], max_sources=max_sources)
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
    input_file_climate = "data/final-data/final_climate_data.json"
    input_file_covid = "data/final-data/final_covid_data.json"
    output_file_climate = "data/final-data/enriched_climate_data_40.json"
    output_file_covid = "data/final-data/enriched_covid_data_40.json"
    api_key = "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"
    
    enrich_data(input_file_climate, output_file_climate, api_key, max_sources=40)
    enrich_data(input_file_covid, output_file_covid, api_key, max_sources=40)
