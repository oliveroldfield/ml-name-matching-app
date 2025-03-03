import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
import jellyfish
import re
from fuzzywuzzy import fuzz
import pickle
import os

def preprocess_name(name):
    """Clean and standardize name strings."""
    if not isinstance(name, str):
        return ""
    name = name.lower()
    name = re.sub(r'[^\w\s]', '', name)
    return name.strip()

def extract_features(name1, name2):
    """Extract comparison features between two names."""
    name1 = preprocess_name(name1)
    name2 = preprocess_name(name2)
    
    features = {
        'jaro_winkler': jellyfish.jaro_winkler(name1, name2),
        'levenshtein': jellyfish.levenshtein_distance(name1, name2),
        'damerau_levenshtein': jellyfish.damerau_levenshtein_distance(name1, name2),
        'hamming': jellyfish.hamming_distance(name1, name2) if len(name1) == len(name2) else min(len(name1), len(name2)),
        'fuzz_ratio': fuzz.ratio(name1, name2) / 100,
        'fuzz_partial': fuzz.partial_ratio(name1, name2) / 100,
        'fuzz_token_sort': fuzz.token_sort_ratio(name1, name2) / 100,
        'fuzz_token_set': fuzz.token_set_ratio(name1, name2) / 100,
        'len_diff': abs(len(name1) - len(name2)),
        'first_char_match': int(name1[0] == name2[0]) if name1 and name2 else 0
    }
    
    return features

def create_feature_df(names_df):
    """Create features dataframe from pairs of names."""
    features_list = []
    
    for _, row in names_df.iterrows():
        features = extract_features(row['name1'], row['name2'])
        features_list.append(features)
    
    return pd.DataFrame(features_list)

def train_model(train_df):
    """Train a RandomForest model on name matching features."""
    X = create_feature_df(train_df)
    y = train_df['match'].astype(int)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model

def save_model(model, path='name_matcher_model.pkl'):
    """Save the trained model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path='name_matcher_model.pkl'):
    """Load a trained model from disk."""
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return None

def evaluate_model(model, test_df):
    """Evaluate model performance on test data."""
    X_test = create_feature_df(test_df)
    y_test = test_df['match'].astype(int)
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }
    
    return metrics, y_pred

def main():
    st.title("ML-Based Name Matching")
    
    tab1, tab2, tab3 = st.tabs(["Train Model", "Test Model", "Match Names"])
    
    with tab1:
        st.header("Train Your Name Matching Model")
        
        st.subheader("Upload Training Data")
        st.write("Upload a CSV with columns: name1, name2, match (1 for match, 0 for non-match)")
        
        train_file = st.file_uploader("Upload training data", type=["csv"])
        
        if train_file is not None:
            train_df = pd.read_csv(train_file)
            st.write(f"Loaded {len(train_df)} training examples")
            st.dataframe(train_df.head())
            
            if st.button("Train Model"):
                with st.spinner("Training model..."):
                    model = train_model(train_df)
                    save_model(model)
                st.success("Model trained and saved!")
                
                # Show feature importance
                X = create_feature_df(train_df)
                feature_importance = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                st.subheader("Feature Importance")
                st.bar_chart(feature_importance.set_index('Feature'))
    
    with tab2:
        st.header("Test Your Model")
        
        model = load_model()
        if model is None:
            st.warning("No trained model found. Please train a model first.")
        else:
            st.success("Model loaded successfully!")
            
            st.subheader("Upload Test Data")
            st.write("Upload a CSV with columns: name1, name2, match (1 for match, 0 for non-match)")
            
            test_file = st.file_uploader("Upload test data", type=["csv"])
            
            if test_file is not None:
                test_df = pd.read_csv(test_file)
                st.write(f"Loaded {len(test_df)} test examples")
                
                if st.button("Evaluate Model"):
                    with st.spinner("Evaluating model..."):
                        metrics, predictions = evaluate_model(model, test_df)
                    
                    st.subheader("Model Performance")
                    metrics_df = pd.DataFrame({
                        'Metric': metrics.keys(),
                        'Value': metrics.values()
                    })
                    st.dataframe(metrics_df)
                    
                    # Show predictions
                    test_df['predicted'] = predictions
                    st.subheader("Predictions")
                    st.dataframe(test_df)
    
    with tab3:
        st.header("Match Names")
        
        model = load_model()
        if model is None:
            st.warning("No trained model found. Please train a model first.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                name1 = st.text_input("Name 1")
            
            with col2:
                name2 = st.text_input("Name 2")
            
            if name1 and name2:
                features = extract_features(name1, name2)
                features_df = pd.DataFrame([features])
                
                prediction = model.predict(features_df)[0]
                probability = model.predict_proba(features_df)[0][1]
                
                st.subheader("Match Prediction")
                if prediction == 1:
                    st.success(f"MATCH with {probability:.2%} confidence")
                else:
                    st.error(f"NOT A MATCH with {1-probability:.2%} confidence")
                
                st.subheader("Feature Values")
                features_display = pd.DataFrame({
                    'Feature': features.keys(),
                    'Value': features.values()
                })
                st.dataframe(features_display)
                
                # Allow bulk matching
                st.subheader("Bulk Matching")
                st.write("Upload a CSV with columns: name1, name2")
                
                bulk_file = st.file_uploader("Upload names to match", type=["csv"])
                
                if bulk_file is not None:
                    bulk_df = pd.read_csv(bulk_file)
                    
                    if st.button("Run Bulk Matching"):
                        with st.spinner("Matching names..."):
                            features_df = create_feature_df(bulk_df)
                            bulk_df['match'] = model.predict(features_df)
                            bulk_df['confidence'] = model.predict_proba(features_df)[:, 1]
                        
                        st.subheader("Matching Results")
                        st.dataframe(bulk_df)
                        
                        csv = bulk_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            "Download Results",
                            csv,
                            "name_matching_results.csv",
                            "text/csv",
                            key='download-csv'
                        )

if __name__ == "__main__":
    main()