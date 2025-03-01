import numpy as np
from sentence_transformers import SentenceTransformer, util
import json

# Load the model
print("Loading sentence transformer model...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Define the questions
CLIMATE_QUESTIONS = [
    "Do you believe climate change is real?",
    "Do you believe human activities are the main cause of climate change?",
    "Do you believe human-caused climate change has increased the frequency of extreme weather events (hurricanes, heatwaves, wildfires) beyond their natural occurrence patterns?",
    "Do you believe scientists are exaggerating climate change impacts like melting ice caps, rising sea levels, and global warming?"
]

COVID_QUESTIONS = [
    "Do you think COVID-19 was a man-made virus?",
    "Do you think COVID-19 vaccines are safe and effective?",
    "Do you think masks provide real protection against COVID-19?",
    "Do you think stay-at-home measures helped prevent the spread of COVID-19?"
]

# Sample climate claims
CLIMATE_CLAIMS = [
    "Climate change is due to cosmic rays",
    "That human CO2 is causing global warming is known with high certainty & confirmed by observations",
    "Global warming is causing more hurricanes and stronger hurricanes",
    "Scientists are exaggerating the impacts of climate change to secure more funding"
]

# Sample COVID claims (hypothetical examples)
COVID_CLAIMS = [
    "COVID-19 was engineered in a lab in Wuhan, China",
    "COVID-19 vaccines have dangerous side effects and were rushed to market without proper testing",
    "Masks are ineffective at preventing the spread of COVID-19 and may actually be harmful",
    "Lockdowns caused more harm than good and had minimal impact on reducing COVID-19 transmission"
]

def categorize_claims(claims, questions):
    """Categorize claims by their semantic similarity to questions, distributing evenly"""
    # Encode questions
    print("Encoding questions...")
    question_embeddings = model.encode(questions)
    
    # Encode all claims
    print(f"Encoding {len(claims)} claims...")
    claim_embeddings = model.encode(claims)
    
    # Calculate similarity matrix between all claims and all questions
    similarity_matrix = util.cos_sim(claim_embeddings, question_embeddings)
    
    # Initialize results dictionary
    results = {q: [] for q in questions}
    
    # Track which claims have been assigned
    assigned_claims = set()
    
    # Calculate how many claims per question (evenly distributed)
    claims_per_question = len(claims) // len(questions)
    remainder = len(claims) % len(questions)
    
    print(f"\nDistributing claims across {len(questions)} questions ({claims_per_question} per question, with {remainder} extra)")
    
    # First pass: assign claims to their most similar questions, respecting the distribution
    for q_idx, question in enumerate(questions):
        # How many claims should this question get
        target_count = claims_per_question + (1 if q_idx < remainder else 0)
        
        # Get all similarities for this question
        q_similarities = similarity_matrix[:, q_idx]
        
        # Create pairs of (claim_idx, similarity)
        claim_sim_pairs = [(i, float(sim)) for i, sim in enumerate(q_similarities) if i not in assigned_claims]
        
        # Sort by similarity (highest first)
        claim_sim_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Take the top N claims for this question
        for i in range(min(target_count, len(claim_sim_pairs))):
            claim_idx, similarity = claim_sim_pairs[i]
            results[question].append({
                "claim": claims[claim_idx],
                "similarity_score": similarity
            })
            assigned_claims.add(claim_idx)
            
            print(f"  Assigned claim: '{claims[claim_idx]}' to question: '{question}' (score: {similarity:.4f})")
    
    # Handle any remaining unassigned claims (should be rare due to the even distribution)
    if len(assigned_claims) < len(claims):
        print("\nAssigning remaining claims:")
        for claim_idx in range(len(claims)):
            if claim_idx not in assigned_claims:
                # Find the best question for this claim
                q_similarities = similarity_matrix[claim_idx]
                best_q_idx = np.argmax(q_similarities)
                best_question = questions[best_q_idx]
                
                results[best_question].append({
                    "claim": claims[claim_idx],
                    "similarity_score": float(q_similarities[best_q_idx])
                })
                
                print(f"  Assigned remaining claim: '{claims[claim_idx]}' to question: '{best_question}' (score: {float(q_similarities[best_q_idx]):.4f})")
    
    return results

def main():
    print("\n=== CLIMATE CLAIMS CATEGORIZATION ===")
    climate_results = categorize_claims(CLIMATE_CLAIMS, CLIMATE_QUESTIONS)
    
    print("\n=== COVID-19 CLAIMS CATEGORIZATION ===")
    covid_results = categorize_claims(COVID_CLAIMS, COVID_QUESTIONS)
    
    # Save results
    with open('demo_results.json', 'w') as f:
        json.dump({
            "climate": climate_results,
            "covid": covid_results
        }, f, indent=2)
    
    print("\nResults saved to demo_results.json")

if __name__ == "__main__":
    main() 