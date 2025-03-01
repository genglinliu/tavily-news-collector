import json
import argparse
import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

# Set up argument parser
parser = argparse.ArgumentParser(description='Categorize COVID-19 claims by their semantic similarity to questions')
parser.add_argument('--claims_file', type=str, required=True, 
                    help='Path to the COVID-19 claims JSONL file')
parser.add_argument('--output_file', type=str, default='covid_claims_categorized.json',
                    help='Path to save the output JSON file')
args = parser.parse_args()

# COVID-19 questions
COVID_QUESTIONS = [
    "Do you think COVID-19 was a man-made virus?",
    "Do you think COVID-19 vaccines are safe and effective?",
    "Do you think masks provide real protection against COVID-19?",
    "Do you think stay-at-home measures helped prevent the spread of COVID-19?"
]

def load_claims(file_path):
    """Load claims from a JSONL file"""
    claims = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                claim_data = json.loads(line)
                claims.append(claim_data['claim'])
    return claims

def categorize_claims(claims, questions):
    """Categorize claims by their semantic similarity to questions"""
    # Load the model
    print("Loading sentence transformer model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Encode questions
    print("Encoding questions...")
    question_embeddings = model.encode(questions)
    
    # Initialize results dictionary
    results = {q: [] for q in questions}
    
    # Process claims
    print(f"Processing {len(claims)} claims...")
    for claim in tqdm(claims):
        # Encode the claim
        claim_embedding = model.encode([claim])
        
        # Calculate similarity with each question
        similarities = util.cos_sim(claim_embedding, question_embeddings)[0]
        
        # Find the question with the highest similarity
        most_similar_idx = np.argmax(similarities)
        most_similar_question = questions[most_similar_idx]
        similarity_score = similarities[most_similar_idx].item()
        
        # Add claim to the corresponding question's list
        results[most_similar_question].append({
            "claim": claim,
            "similarity_score": similarity_score
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
    categorized_claims = categorize_claims(claims, COVID_QUESTIONS)
    
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
    
    # Print distribution
    print("\nDistribution of claims across questions:")
    for question, claims_list in categorized_claims.items():
        print(f"  {question}: {len(claims_list)} claims")

if __name__ == "__main__":
    main() 