import json
import re
import argparse
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up argument parser
parser = argparse.ArgumentParser(description='Group claims by their semantic similarity to questions')
parser.add_argument('--claims_file', type=str, default='claims_climate.jsonl', 
                    help='Path to the claims JSONL file')
parser.add_argument('--output_file', type=str, default='claims_grouped_by_questions_simple.json',
                    help='Path to save the output JSON file')
args = parser.parse_args()

# Download NLTK resources (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Climate questions
CLIMATE_QUESTIONS = [
    "Do you believe climate change is real?",
    "Do you believe human activities are the main cause of climate change?",
    "Do you believe human-caused climate change has increased the frequency of extreme weather events (hurricanes, heatwaves, wildfires) beyond their natural occurrence patterns?",
    "Do you believe scientists are exaggerating climate change impacts like melting ice caps, rising sea levels, and global warming?"
]

# Keywords for each question to improve matching
QUESTION_KEYWORDS = {
    CLIMATE_QUESTIONS[0]: [
        "climate change", "global warming", "real", "exists", "happening", 
        "evidence", "data", "temperature", "warming", "climate", "trend"
    ],
    CLIMATE_QUESTIONS[1]: [
        "human", "anthropogenic", "man-made", "cause", "CO2", "carbon dioxide", 
        "greenhouse gas", "emissions", "human activity", "industrial", "fossil fuel"
    ],
    CLIMATE_QUESTIONS[2]: [
        "extreme weather", "hurricane", "heatwave", "wildfire", "drought", "flood", 
        "storm", "natural disaster", "frequency", "intensity", "severe", "pattern"
    ],
    CLIMATE_QUESTIONS[3]: [
        "exaggerate", "overstate", "alarmist", "scientist", "prediction", "model", 
        "ice cap", "sea level", "melting", "rising", "impact", "projection", "IPCC"
    ]
}

def preprocess_text(text):
    """Preprocess text by removing stopwords, lemmatizing, etc."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return ' '.join(tokens)

def load_claims(file_path):
    """Load claims from a JSONL file"""
    claims = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                claim_data = json.loads(line)
                claims.append(claim_data['claim'])
    return claims

def categorize_claims_tfidf(claims, questions, question_keywords):
    """Categorize claims using TF-IDF and cosine similarity"""
    # Preprocess claims and questions
    preprocessed_claims = [preprocess_text(claim) for claim in claims]
    
    # Create augmented question texts with keywords
    augmented_questions = []
    for q in questions:
        keywords = question_keywords[q]
        augmented_q = q + " " + " ".join(keywords)
        augmented_questions.append(preprocess_text(augmented_q))
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    
    # Fit and transform on all texts
    all_texts = preprocessed_claims + augmented_questions
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split the matrix back into claims and questions
    claims_tfidf = tfidf_matrix[:len(preprocessed_claims)]
    questions_tfidf = tfidf_matrix[len(preprocessed_claims):]
    
    # Calculate similarity between each claim and each question
    similarity_matrix = cosine_similarity(claims_tfidf, questions_tfidf)
    
    # Initialize results dictionary
    results = {q: [] for q in questions}
    
    # Assign each claim to the most similar question
    for i, claim in enumerate(claims):
        # Find the question with the highest similarity
        most_similar_idx = np.argmax(similarity_matrix[i])
        most_similar_question = questions[most_similar_idx]
        similarity_score = similarity_matrix[i][most_similar_idx]
        
        # Add claim to the corresponding question's list
        results[most_similar_question].append({
            "claim": claim,
            "similarity_score": float(similarity_score)
        })
    
    # Sort claims by similarity score within each question
    for question in results:
        results[question] = sorted(results[question], key=lambda x: x['similarity_score'], reverse=True)
    
    return results

def main():
    # Load claims
    claims_path = args.claims_file
    print(f"Loading claims from {claims_path}...")
    claims = load_claims(claims_path)
    
    # Categorize claims
    print("Categorizing claims...")
    categorized_claims = categorize_claims_tfidf(claims, CLIMATE_QUESTIONS, QUESTION_KEYWORDS)
    
    # Save results
    output_path = args.output_file
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(categorized_claims, f, indent=2)
    
    # Print summary
    print("\nSummary:")
    for question, claims_list in categorized_claims.items():
        print(f"\n{question}: {len(claims_list)} claims")
        # Print top 5 claims for each question
        for i, claim_data in enumerate(claims_list[:5]):
            print(f"  {i+1}. {claim_data['claim']} (score: {claim_data['similarity_score']:.4f})")
    
    # Print distribution
    print("\nDistribution of claims across questions:")
    for question, claims_list in categorized_claims.items():
        print(f"  {question}: {len(claims_list)} claims")

if __name__ == "__main__":
    main() 