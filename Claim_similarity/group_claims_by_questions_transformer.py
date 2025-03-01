import json
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set up argument parser
parser = argparse.ArgumentParser(description='Group claims by their semantic similarity to questions using sentence transformers')
parser.add_argument('--claims_file', type=str, default='claims_climate.jsonl', 
                    help='Path to the claims JSONL file')
parser.add_argument('--output_file', type=str, default='claims_grouped_by_questions_transformer.json',
                    help='Path to save the output JSON file')
parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                    help='Sentence transformer model to use (default: all-MiniLM-L6-v2)')
args = parser.parse_args()

# Climate questions
CLIMATE_QUESTIONS = [
    "Do you believe climate change is real?",
    "Do you believe human activities are the main cause of climate change?",
    "Do you believe human-caused climate change has increased the frequency of extreme weather events (hurricanes, heatwaves, wildfires) beyond their natural occurrence patterns?",
    "Do you believe scientists are exaggerating climate change impacts like melting ice caps, rising sea levels, and global warming?"
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

def categorize_claims_transformer(claims, questions, model_name):
    """Categorize claims using sentence transformers and cosine similarity"""
    print(f"Loading sentence transformer model: {model_name}")
    model = SentenceTransformer(model_name)
    
    print("Encoding questions...")
    question_embeddings = model.encode(questions, show_progress_bar=True)
    
    # Initialize results dictionary
    results = {q: [] for q in questions}
    
    print(f"Processing {len(claims)} claims...")
    # Process claims in batches to be more efficient
    batch_size = 32
    for i in tqdm(range(0, len(claims), batch_size)):
        batch_claims = claims[i:i+batch_size]
        
        # Encode batch of claims
        batch_embeddings = model.encode(batch_claims, show_progress_bar=False)
        
        # Calculate similarity with each question for each claim in the batch
        batch_similarities = cosine_similarity(batch_embeddings, question_embeddings)
        
        # Assign each claim to the most similar question
        for j, claim in enumerate(batch_claims):
            # Find the question with the highest similarity
            most_similar_idx = np.argmax(batch_similarities[j])
            most_similar_question = questions[most_similar_idx]
            similarity_score = batch_similarities[j][most_similar_idx]
            
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
    categorized_claims = categorize_claims_transformer(claims, CLIMATE_QUESTIONS, args.model)
    
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
    
    # Calculate average similarity scores for each question
    print("\nAverage similarity scores:")
    for question, claims_list in categorized_claims.items():
        avg_score = sum(item['similarity_score'] for item in claims_list) / len(claims_list) if claims_list else 0
        print(f"  {question}: {avg_score:.4f}")

if __name__ == "__main__":
    main() 