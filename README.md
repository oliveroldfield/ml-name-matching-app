# Name Matching ML Demo

This app uses machine learning to perform fuzzy name matching, learning from examples of matching and non-matching names.

## Setup

1. Install dependencies:
```
pip install streamlit pandas numpy scikit-learn jellyfish fuzzywuzzy python-Levenshtein
```

2. Run the app:
```
streamlit run streamlit_app.py
```

## Features

- **Train** a machine learning model using your own positive and negative name match examples
- **Test** your model with validation data to see accuracy metrics
- **Match** names interactively or in bulk
- View detailed matching features and confidence scores

## How It Works

The app extracts multiple similarity features between name pairs:
- Jaro-Winkler distance
- Levenshtein distance
- Damerau-Levenshtein distance
- Hamming distance
- Fuzzy string matching ratios
- Character-level metrics

These features feed into a Random Forest classifier that learns to predict matches based on training data.

## Data Format

Training and test data should be CSV files with these columns:
- `name1`: First name to compare
- `name2`: Second name to compare
- `match`: 1 for matching names, 0 for non-matching names

## Sample Data

Use the included `sample_data_generator.py` script to create sample data for testing:

```
python sample_data_generator.py
```

This generates:
- `name_matching_train.csv`: 800 sample name pairs for training
- `name_matching_test.csv`: 200 sample name pairs for testing

## Tips for Best Results

1. Provide diverse examples of both matching and non-matching name pairs
2. Include examples with common errors like typos, abbreviations, and name order variations
3. Balance your dataset with similar numbers of matches and non-matches
4. Experiment with the confidence threshold depending on your use case (higher for precision, lower for recall)