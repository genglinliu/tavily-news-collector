# Claim Similarity Categorization

This repository contains scripts to categorize climate-related claims based on their semantic similarity to screening questions used in a human study.

## Overview

The goal is to categorize climate claims based on their semantic similarity to four screening questions:

1. "Do you believe climate change is real?"
2. "Do you believe human activities are the main cause of climate change?"
3. "Do you believe human-caused climate change has increased the frequency of extreme weather events (hurricanes, heatwaves, wildfires) beyond their natural occurrence patterns?"
4. "Do you believe scientists are exaggerating climate change impacts like melting ice caps, rising sea levels, and global warming?"

We provide three different implementations with varying levels of sophistication:

1. `group_claims_by_questions.py` - Uses OpenAI's API for high-quality embeddings
2. `group_claims_by_questions_simple.py` - Uses TF-IDF and basic NLP techniques (no API required)
3. `group_claims_by_questions_transformer.py` - Uses sentence-transformers for better semantic matching (no API required)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd Claim_similarity

# Install dependencies
pip install -r requirements.txt

# For the simple version, you may need to download NLTK resources
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Requirements

Create a `requirements.txt` file with:

```
numpy
scikit-learn
pandas
tqdm
nltk
openai  # Only needed for the OpenAI API version
sentence-transformers  # Only needed for the transformer version
```

## Usage

### OpenAI API Version

```bash
python group_claims_by_questions.py --claims_file claims_climate.jsonl --output_file results_openai.json --api_key YOUR_OPENAI_API_KEY
```

### Simple Version (TF-IDF)

```bash
python group_claims_by_questions_simple.py --claims_file claims_climate.jsonl --output_file results_simple.json
```

### Transformer Version

```bash
python group_claims_by_questions_transformer.py --claims_file claims_climate.jsonl --output_file results_transformer.json
```

You can specify a different sentence transformer model:

```bash
python group_claims_by_questions_transformer.py --model all-mpnet-base-v2 --claims_file claims_climate.jsonl --output_file results_transformer_mpnet.json
```

## Output Format

The output is a JSON file with the following structure:

```json
{
  "Do you believe climate change is real?": [
    {
      "claim": "Climate change is due to cosmic rays",
      "similarity_score": 0.8765
    },
    ...
  ],
  "Do you believe human activities are the main cause of climate change?": [
    ...
  ],
  ...
}
```

Claims are sorted by similarity score within each question category.

## Recommended Approach

For the best results, we recommend using the transformer-based approach (`group_claims_by_questions_transformer.py`) as it provides a good balance between accuracy and ease of use without requiring an API key.

If you need the highest quality results and have access to an OpenAI API key, use the OpenAI version (`group_claims_by_questions.py`).

## Example Results

Here's an example of what the top claims might look like for each question:

### Question 1: "Do you believe climate change is real?"
1. "Climate change is due to cosmic rays" (score: 0.8765)
2. "The geologic record provides us with abundant evidence for such perpetual natural climate variability..." (score: 0.8543)
3. "Actual weather records over the past 100 years show no correlation between rising carbon dioxide levels and local temperatures" (score: 0.8321)

### Question 2: "Do you believe human activities are the main cause of climate change?"
1. "That human CO2 is causing global warming is known with high certainty & confirmed by observations" (score: 0.9123)
2. "Theory, models and direct measurement confirm CO2 is currently the main driver of climate change" (score: 0.8976)
3. "Human emissions upset the natural balance, rising CO2 to levels not seen in at least 800,000 years" (score: 0.8765)

And so on for the other questions. 