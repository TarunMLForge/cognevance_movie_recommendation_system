import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

def build_user_item_matrix(df):
    """
    Creates a user-item rating matrix from the processed dataset.
    """
    print("Building user-item matrix...")
    matrix = df.pivot(index='userId', columns='title', values='rating')
    # Fill NaN with 0 for Cosine Similarity calculation
    matrix.fillna(0, inplace=True)
    return matrix

def calculate_item_similarity(user_item_matrix):
    """
    Calculates item-item similarity using Cosine Similarity.
    """
    print("Calculating item-item similarity...")
    # Transpose matrix to compute similarity between items (movies) instead of users
    item_similarity = cosine_similarity(user_item_matrix.T)
    
    # Wrap in a DataFrame for easy querying by movie title
    item_similarity_df = pd.DataFrame(
        item_similarity, 
        index=user_item_matrix.columns, 
        columns=user_item_matrix.columns
    )
    return item_similarity_df

def get_movie_recommendations(movie_title, item_similarity_df, top_n=5):
    """
    Recommends top N movies similar to a given movie.
    """
    if movie_title not in item_similarity_df.columns:
        return f"Movie '{movie_title}' not found in the dataset."
        
    print(f"Generating recommendations based on: '{movie_title}'")
    # Get similar movies and sort descending
    similar_movies = item_similarity_df[movie_title].sort_values(ascending=False)
    
    # Remove the queried movie itself from the list
    similar_movies = similar_movies.drop(movie_title)
    
    return similar_movies.head(top_n)

def save_model(item_similarity_df, output_path="models/item_similarity_model.pkl"):
    """
    Saves the similarity matrix to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(item_similarity_df, f)
    print(f"Model saved to {output_path}")

def load_model(input_path="models/item_similarity_model.pkl"):
    """
    Loads the saved similarity matrix.
    """
    with open(input_path, 'rb') as f:
        return pickle.load(f)

if __name__ == "__main__":
    is_src = os.path.basename(os.getcwd()) == "src"
    data_path = "../data/processed/movie_ratings_processed.csv" if is_src else "data/processed/movie_ratings_processed.csv"
    model_path = "../models/item_similarity_model.pkl" if is_src else "models/item_similarity_model.pkl"
    
    # Load data
    df = pd.read_csv(data_path)
    
    # Build Model
    user_item_matrix = build_user_item_matrix(df)
    item_similarity_df = calculate_item_similarity(user_item_matrix)
    
    # Save Model
    save_model(item_similarity_df, model_path)
    
    # Test recommendations
    sample_movie = "Matrix, The (1999)"
    recommendations = get_movie_recommendations(sample_movie, item_similarity_df)
    print(f"\nTop 5 Recommendations for users who liked {sample_movie}:")
    print(recommendations)
