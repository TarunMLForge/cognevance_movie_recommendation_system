import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
import os
import sys

from src.recommender import build_user_item_matrix, calculate_item_similarity

def evaluate_model(df):
    """
    Evaluates the collaborative filtering model using Root Mean Squared Error (RMSE).
    Splits the dataset into 80% train and 20% test to prevent data leakage.
    """
    print("Splitting dataset into train and test sets (80/20)...")
    train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)
    
    # Build model using ONLY training data
    user_item_matrix = build_user_item_matrix(train_data)
    item_similarity_df = calculate_item_similarity(user_item_matrix)
    
    print("Evaluating predictions on the test set...")
    y_true = []
    y_pred = []
    
    # Convert item_similarity_df to numpy array for faster lookups
    item_sim_mat = item_similarity_df.values
    movies_list = item_similarity_df.columns.tolist()
    movie_to_idx = {movie: idx for idx, movie in enumerate(movies_list)}
    
    # Iterate through test data to predict ratings
    for idx, row in test_data.iterrows():
        user = row['userId']
        movie = row['title']
        actual_rating = row['rating']
        
        # If the movie or user isn't in our training matrix (cold start problem), skip
        if movie not in movie_to_idx or user not in user_item_matrix.index:
            continue
            
        movie_idx = movie_to_idx[movie]
        
        # Get user's ratings for all movies
        user_ratings = user_item_matrix.loc[user].values
        
        # Get similarities for the target movie
        movie_similarities = item_sim_mat[movie_idx]
        
        # Calculate predicted rating: weighted average of user's ratings by movie similarity
        # Only consider movies the user has actually rated (>0)
        rated_indices = np.where(user_ratings > 0)[0]
        
        if len(rated_indices) == 0:
            continue
            
        # Extract similarities and ratings for items the user has rated
        sim_scores = movie_similarities[rated_indices]
        ratings = user_ratings[rated_indices]
        
        # Avoid division by zero and negative similarities
        valid_mask = sim_scores > 0
        if np.sum(sim_scores[valid_mask]) > 0:
            predicted_rating = np.dot(sim_scores[valid_mask], ratings[valid_mask]) / np.sum(sim_scores[valid_mask])
            
            # Clip prediction to the 0.5 - 5.0 scale
            predicted_rating = max(0.5, min(5.0, predicted_rating))
            
            y_true.append(actual_rating)
            y_pred.append(predicted_rating)
            
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    print(f"\n================ Evaluation Results ================")
    print(f"Total Test Samples Evaluated: {len(y_true)}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print("====================================================\n")
    
    return rmse

if __name__ == "__main__":
    is_src = os.path.basename(os.getcwd()) == "src"
    data_path = "../data/processed/movie_ratings_processed.csv" if is_src else "data/processed/movie_ratings_processed.csv"
    
    df = pd.read_csv(data_path)
    evaluate_model(df)
