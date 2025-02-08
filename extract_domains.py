import pandas as pd

def extract_untrusted_domains(csv_path, min_score=0, max_score=50):
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
        (df['Score'] <= max_score) &
        (df['Country'] == 'US') &
        (df['Language'] == 'en')
    ][['Domain', 'Score']].sort_values('Score')
    
    # Remove duplicates (keeping first occurrence which will have lowest score)
    filtered_domains = filtered_domains.drop_duplicates('Domain')
    
    # Convert to list
    return filtered_domains['Domain'].tolist()

# Example usage:
domains = extract_untrusted_domains("NewsGuard/label-full-metadata-20241219.csv")
print(f"Found {len(domains)} domains")
print(domains)