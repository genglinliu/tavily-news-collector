import pandas as pd
import os
from typing import Dict, Any
from tavily import TavilyClient
import json
from tqdm import tqdm
import random
from domains import fake_news_domains_newsguard, fake_news_domains_wiki
from tenacity import retry, stop_after_attempt, wait_exponential
FAKE_NEWS_DOMAINS = random.sample(fake_news_domains_newsguard + fake_news_domains_wiki, 600)
# FAKE_NEWS_DOMAINS = fake_news_domains_newsguard + fake_news_domains_wiki

MAX_SOURCES = 15

class NewsSearch:

    def __init__(self, api_key: str = None):
        """Initialize with Tavily API key from environment variable or parameter."""
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Please provide TAVILY_API_KEY as environment variable or parameter")
        self.client = TavilyClient(api_key=self.api_key)

    @retry(stop=stop_after_attempt(1000), wait=wait_exponential(multiplier=1, min=4, max=15))
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
            include_answer=True,
            include_raw_content=True,
        )


def enrich_claim_with_sources(searcher: NewsSearch, claim: str, output_file: str, item: dict):
    """
    Enrich a claim with supporting and opposing sources by performing searches and append to output file immediately.
    
    Args:
        searcher: NewsSearch instance
        claim: The claim to search for
        output_file: Path to output file to append results
        item: The full item dictionary containing the claim
    """
    item.update({
        "supporting_sources": [],
        "opposing_sources": []
    })
    
    seen_supporting_urls = set()
    seen_opposing_urls = set()
    
    # Try a few different search variations to get more diverse results
    search_variations = [
        claim,
        f"Help me find content regarding this claim: {claim}",
        f"Find sources about this claim: {claim}",
        f"Research the claim: {claim}"
    ]
    
    for query in search_variations:
        if len(item["supporting_sources"]) >= MAX_SOURCES:
            break
            
        results = searcher.search(query, include_domains=None)
        
        # Add new unique results
        for result in results["results"]:
            if result["url"] not in seen_supporting_urls and len(item["supporting_sources"]) < MAX_SOURCES:
                source = {
                    "title": result["title"],
                    "url": result["url"],
                    "content": result["content"],
                    "raw_content": result["raw_content"]
                }
                item["supporting_sources"].append(source)
                seen_supporting_urls.add(result["url"])
                print(f"Added supporting source: {result['url']}")
                
                # Append to file after each new source
                with open(output_file, 'r+') as f:
                    data = json.load(f)
                    f.seek(0)
                    json.dump(data[:-1] + [item], f, indent=4)
    
    print(f"Found {len(item['supporting_sources'])} supporting sources")    
    
    # Collect opposing sources through multiple searches
    domains_to_try = FAKE_NEWS_DOMAINS.copy()
    
    while len(item["opposing_sources"]) < MAX_SOURCES and domains_to_try:
        current_domains = domains_to_try[:200]
        domains_to_try = domains_to_try[200:]
        print(len(domains_to_try))
        
        misinfo_results = searcher.search(claim, include_domains=current_domains)
        
        for result in misinfo_results["results"]:
            if result["url"] not in seen_opposing_urls and len(item["opposing_sources"]) < MAX_SOURCES:
                source = {
                    "title": result["title"],
                    "url": result["url"],
                    "content": result["content"],
                    "raw_content": result["raw_content"]
                }
                item["opposing_sources"].append(source)
                seen_opposing_urls.add(result["url"])
                print(f"Added opposing source: {result['url']}")
                
                # Append to file after each new source
                with open(output_file, 'r+') as f:
                    data = json.load(f)
                    f.seek(0)
                    json.dump(data[:-1] + [item], f, indent=4)
                
        print(f"Found {len(misinfo_results['results'])} opposing sources, trying again")
                
    print(f"Found {len(item['opposing_sources'])} opposing sources")

def enrich_data(input_file: str, output_file: str, api_key: str = None):
    """
    Read COVID data, enrich with sources, and append to output file as we go.
    Skip claims that have already been processed in the output file.
    """
    searcher = NewsSearch(api_key=api_key)
    
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    # data = data[:50]
    
    # Check for existing claims in output file
    processed_claims = set()
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                existing_data = json.load(f)
                processed_claims = {item['claim'] for item in existing_data}
        except json.JSONDecodeError:
            existing_data = []
            # Create new file with empty array
            with open(output_file, 'w') as f:
                json.dump([], f)
    else:
        existing_data = []
        # Create new file with empty array
        with open(output_file, 'w') as f:
            json.dump([], f)
    
    # Filter out claims that have already been processed
    items_to_process = [item for item in data if item['claim'] not in processed_claims]
    print(f"Found {len(existing_data)} existing processed claims")
    print(f"Processing {len(items_to_process)} new claims")
    
    for item in tqdm(items_to_process):
        try:
            # Add empty item to output file first
            with open(output_file, 'r+') as f:
                data = json.load(f)
                data.append(item)
                f.seek(0)
                json.dump(data, f, indent=4)
            
            # Now enrich it with sources
            enrich_claim_with_sources(searcher, item["claim"], output_file, item)
            
        except Exception as e:
            print(f"\nError processing claim: {item['claim']}")
            print(f"Error: {str(e)}")
            raise e
    
    print("\nCompleted processing all items")

if __name__ == "__main__":
    # Example usage
    input_file_climate = "data/final-data/final_climate_data.json"
    input_file_covid = "data/final-data/final_covid_data.json"
    output_file_climate = "data/final-data/enriched_climate_data_15.json"
    output_file_covid = "data/final-data/enriched_covid_data_15.json"
    # api_key = "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"
    api_key = "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"
    
    enrich_data(input_file_climate, output_file_climate, api_key)
    enrich_data(input_file_covid, output_file_covid, api_key)
