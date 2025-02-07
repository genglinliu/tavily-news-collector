# import asyncio
# from typing import Dict, Any
# from tavily import TavilyClient
# from gpt_researcher import GPTResearcher

# class TavilyResearch:
#     def __init__(self, api_key: str):
#         self.client = TavilyClient(api_key=api_key)
        
#     def basic_search(self, query: str) -> Dict[str, Any]:
#         """Basic search with Tavily's search API"""
#         return self.client.search(
#             query,
#             #search_depth="advanced",  # Using advanced for more thorough results
#             # include_answer=True,  # Get a summarized answer
#             max_results=10,  # Limiting to top 5 results for clarity
#             include_raw_content=True  # Include full parsed content when available
#         )
    
#     def get_context_search(self, query: str, max_tokens: int = 4000) -> str:
#         """Get RAG-optimized context for the query"""
#         return self.client.get_search_context(
#             query,
#             max_tokens=max_tokens,
#             search_depth="advanced"
#         )
    
#     def quick_qa_search(self, query: str) -> str:
#         """Get a quick answer using Tavily's QA search"""
#         return self.client.qna_search(
#             query,
#             search_depth="advanced"
#         )

# async def gpt_researcher_analysis(query: str) -> str:
#     """Use GPT Researcher for comprehensive analysis"""
#     researcher = GPTResearcher(
#         query=f"Analyze whether this statement is factual or misinformation: '{query}'. "\
#               "Provide scientific evidence and research papers when available.",
#         report_type="research_report"
#     )
#     await researcher.conduct_research()
#     return await researcher.write_report()

# # async def main():
# #     # Get API key from environment variable
# #     import os
# #     api_key = os.getenv('TAVILY_API_KEY')
# #     if not api_key:
# #         raise ValueError("Please set TAVILY_API_KEY environment variable")
# #     research = TavilyResearch(api_key)
    
# #     # The statement to analyze
# #     statement = "Ultraviolet light is associated with higher covid-19 growth rates"
    
# #     # 1. Basic Search
# #     print("\n=== Basic Search Results ===")
# #     basic_results = research.basic_search(
# #         f"Is it true that {statement}? Scientific evidence"
# #     )
# #     print(f"Answer: {basic_results.get('answer')}")
# #     print("\nTop Sources:")
# #     for result in basic_results['results']:
# #         print(f"- {result['title']}: {result['url']}")
    
#     # # 2. Context Search (RAG-optimized)
#     # print("\n=== Context Search Results ===")
#     # context = research.get_context_search(
#     #     f"What does scientific research say about whether {statement}?"
#     # )
#     # print(context)
    
#     # # 3. Quick QA Search
#     # print("\n=== Quick QA Results ===")
#     # qa_answer = research.quick_qa_search(
#     #     f"Based on scientific evidence, is it true that {statement}?"
#     # )
#     # print(qa_answer)
    
#     # # 4. GPT Researcher Comprehensive Analysis
#     # print("\n=== GPT Researcher Analysis ===")
#     # research_report = await gpt_researcher_analysis(statement)
#     # print(research_report)

# async def main():
#     # Get API key from environment variable
#     import os
    
#     api_key = os.getenv('TAVILY_API_KEY')
#     if not api_key:
#         print("WARNING: TAVILY_API_KEY not found in environment variables")
#         raise ValueError("Please set TAVILY_API_KEY environment variable")
#     research = TavilyResearch(api_key)
    
#     # The statement to analyze
#     statement = "Ultraviolet light is associated with higher covid-19 growth rates"
    
#     # 1. Basic Search
#     print("\n=== Basic Search Results ===")
#     basic_results = research.basic_search(
#         f"Fake news realted to: {statement}"
#     )
#     print("\nSearch Results:")
#     for result in basic_results['results']:
#         print(f"\nTitle: {result['title']}")
#         print(f"URL: {result['url']}")
#         print(f"Content: {result['content']}\n")



# if __name__ == "__main__":
#     asyncio.run(main())

import pandas as pd

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

# Example usage:
domains = extract_untrusted_domains("label-full-metadata-20241219.csv")


import os
from typing import Dict, Any
from tavily import TavilyClient

class NewsSearch:

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
            include_domains=self.PRIORITY_DOMAINS,
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

def main():
    # Initialize search
    searcher = NewsSearch(api_key="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")
    
    # Your search query
    query = "Ultraviolet light is associated with higher covid-19 growth rates"
    
    # Perform search and print results
    results = searcher.search(query)
    print_results(results)

if __name__ == "__main__":
    main()



misinfo_domains = [
    "naturalnews.com",
    "healthimpactnews.com", 
    "greenmedinfo.com",
    "mercola.com",
    "healthnutnews.com",
    "realfarmacy.com",
    "healthyholisticliving.com",
    "newstarget.com",
    "featureremedies.com",
    "medicalkidnap.com",
    "infowars.com",
    "wnd.com",
    "principia-scientific.com",
    "collective-evolution.com",
    "childrenshealthdefense.org",
    "truthkings.com",
    "goop.com",
    "foodbabe.com",
    "thetruthaboutcancer.com",
    "wakingtimes.com",
    "technocracy.news",
    "vaccineimpact.com",
    "stopmandatoryvaccination.com",
    "beforeitsnews.com",
    "bannedinfo.com",
    "preventdisease.com",
    "naturalsociety.com",
    "alternativemediasyndicate.com",
    "holistichealth.com",
    "prisonplanet.com",
    "banned.video",
    "brighteon.com",
    "yournewswire.com",
    "thepeoplesvoice.tv",
    "neonnettle.com",
    "newspunch.com",
    "americanews.com",
    "conservative101.com",
    "liberalsociety.com",
    "conservativebeaver.com",
    "torontotoday.net",
    "vancouvertimes.org",
    "denverguardian.com",
    "dailystormer.com",
    "gellerreport.com",
    "hoggwatch.com",
    "nationalfile.com",
    "trunews.com",
    "ancient-code.com",
    "dineal.com",
    "ewao.com",
    "galacticconnection.com",
    "geoengineeringwatch.org",
    "in5d.com",
    "responsibletechnology.org",
    "naturalblaze.com",
    "ripostelaique.com",
    "sciencevibe.com",
    "theasociatedpress.com",
    "cbs-news.us",
    "channel23news.com",
    "dailyviralbuzz.com",
    "now8news.com",
    "abcnews-us.com",
    "cnn-globalnews.com",
    "foxnews-us.com",
    "nbcnews11.com",
    "tmzbreaking.com",
    "viralspeech.com",
    "politicsfocus.com",
    "chicksonright.com",
    "climatedepot.com",
    "dailywire.com"
  ]