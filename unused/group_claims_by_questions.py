import json
import os
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tqdm import tqdm
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Group claims by their semantic similarity to questions')
parser.add_argument('--claims_file', type=str, default='claims_climate.jsonl', 
                    help='Path to the claims JSONL file')
parser.add_argument('--output_file', type=str, default='claims_grouped_by_questions.json',
                    help='Path to save the output JSON file')
parser.add_argument('--api_key', type=str, required=True,
                    help='OpenAI API key')
args = parser.parse_args()

# Set up OpenAI API
openai.api_key = args.api_key

# Climate questions
CLIMATE_QUESTIONS = [
    "Do you believe climate change is real?",
    "Do you believe human activities are the main cause of climate change?",
    "Do you believe human-caused climate change has increased the frequency of extreme weather events (hurricanes, heatwaves, wildfires) beyond their natural occurrence patterns?",
    "Do you believe scientists are exaggerating climate change impacts like melting ice caps, rising sea levels, and global warming?"
]

def get_embedding(text, model="text-embedding-ada-002"):
    """Get embedding for a text using OpenAI's API"""
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def load_claims(file_path):
    """Load claims from a JSONL file"""
    claims = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                claim_data = json.loads(line)
                claims.append(claim_data['claim'])
    return claims

def categorize_claims_by_similarity(claims, questions):
    """Categorize claims by their semantic similarity to questions"""
    print("Getting embeddings for questions...")
    question_embeddings = [get_embedding(q) for q in questions]
    
    # Initialize results dictionary
    results = {q: [] for q in questions}
    
    print(f"Processing {len(claims)} claims...")
    for claim in tqdm(claims):
        # Get embedding for the claim
        claim_embedding = get_embedding(claim)
        
        # Calculate similarity with each question
        similarities = []
        for q_embedding in question_embeddings:
            # Convert embeddings to numpy arrays
            q_emb_np = np.array(q_embedding).reshape(1, -1)
            claim_emb_np = np.array(claim_embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            similarity = cosine_similarity(q_emb_np, claim_emb_np)[0][0]
            similarities.append(similarity)
        
        # Find the question with the highest similarity
        most_similar_idx = np.argmax(similarities)
        most_similar_question = questions[most_similar_idx]
        
        # Add claim to the corresponding question's list
        results[most_similar_question].append({
            "claim": claim,
            "similarity_score": float(similarities[most_similar_idx])
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
    categorized_claims = categorize_claims_by_similarity(claims, CLIMATE_QUESTIONS)
    
    # Save results
    output_path = args.output_file
    print(f"Saving results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(categorized_claims, f, indent=2)
    
    # Print summary
    print("\nSummary:")
    for question, claims_list in categorized_claims.items():
        print(f"\n{question}: {len(claims_list)} claims")
        # Print top 3 claims for each question
        for i, claim_data in enumerate(claims_list[:3]):
            print(f"  {i+1}. {claim_data['claim']} (score: {claim_data['similarity_score']:.4f})")

if __name__ == "__main__":
    main()
