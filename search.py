import pandas as pd
import os
from typing import Dict, Any
from tavily import TavilyClient

def extract_untrusted_domains(csv_path, min_score=10, max_score=30):
    """
    Extract domains from NewsGuard CSV that are rated as untrusted ('N') 
    within a specified score range.
    
    Args:
        csv_path (str): Path to the NewsGuard CSV file
        min_score (int): Minimum score threshold (default: 10)
        max_score (int): Maximum score threshold (default: 30)
    
    Returns:
        list: List of domain names meeting the criteria
    """
    # Read the CSV
    df = pd.read_csv(csv_path)
    
    # Filter domains: Rating 'N' and scores within range
    filtered_domains = df[
        (df['Rating'] == 'N') & 
        (df['Score'] >= min_score) & 
        (df['Score'] <= max_score)
    ][['Domain', 'Score']].sort_values('Score')
    
    # Remove duplicates (keeping first occurrence which will have lowest score)
    filtered_domains = filtered_domains.drop_duplicates('Domain')
    
    # Convert to list
    return filtered_domains['Domain'].tolist()


fake_news_domains_wiki = [
    "naturalnews.com", "healthimpactnews.com", "greenmedinfo.com", "mercola.com",
    "healthnutnews.com", "realfarmacy.com", "healthyholisticliving.com", "newstarget.com",
    "featureremedies.com", "medicalkidnap.com", "infowars.com", "wnd.com",
    "principia-scientific.com", "collective-evolution.com", "childrenshealthdefense.org",
    "truthkings.com", "goop.com", "foodbabe.com", "thetruthaboutcancer.com",
    "wakingtimes.com", "technocracy.news", "vaccineimpact.com",
    "stopmandatoryvaccination.com", "beforeitsnews.com", "bannedinfo.com",
    "preventdisease.com", "naturalsociety.com", "alternativemediasyndicate.com",
    "holistichealth.com", "prisonplanet.com", "banned.video", "brighteon.com",
    "yournewswire.com", "thepeoplesvoice.tv", "neonnettle.com", "newspunch.com",
    "americanews.com", "conservative101.com", "liberalsociety.com",
    "conservativebeaver.com", "torontotoday.net", "vancouvertimes.org",
    "denverguardian.com", "dailystormer.com", "gellerreport.com", "hoggwatch.com",
    "nationalfile.com", "trunews.com", "ancient-code.com", "dineal.com", "ewao.com",
    "galacticconnection.com", "geoengineeringwatch.org", "in5d.com",
    "responsibletechnology.org", "naturalblaze.com", "ripostelaique.com",
    "sciencevibe.com", "theasociatedpress.com", "cbs-news.us", "channel23news.com",
    "dailyviralbuzz.com", "now8news.com", "abcnews-us.com", "cnn-globalnews.com",
    "foxnews-us.com", "nbcnews11.com", "tmzbreaking.com", "viralspeech.com",
    "politicsfocus.com", "chicksonright.com", "climatedepot.com", "dailywire.com"
]

fake_news_domains_newsguard = extract_untrusted_domains("label-full-metadata-20241219.csv")

FAKE_NEWS_DOMAINS = fake_news_domains_newsguard + fake_news_domains_wiki 

class NewsSearch:

    def __init__(self, api_key: str = None):
        """Initialize with Tavily API key from environment variable or parameter."""
        self.api_key = api_key or os.getenv('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError("Please provide TAVILY_API_KEY as environment variable or parameter")
        self.client = TavilyClient(api_key=self.api_key)

    def search(self, query: str) -> Dict[str, Any]:
        """
        Perform a search prioritizing specific domains.
        
        Args:
            query: The search query string
            
        Returns:
            Dict containing search results
        """
        return self.client.search(
            query=query,
            include_domains=FAKE_NEWS_DOMAINS,
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


if __name__ == "__main__":
    # Initialize search
    searcher = NewsSearch(api_key="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")
    
    # Your search query
    query = "Ultraviolet light is associated with higher covid-19 growth rates"
    
    # Perform search and print results
    results = searcher.search(query)
    print_results(results)
