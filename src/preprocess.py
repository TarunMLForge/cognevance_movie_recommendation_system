import pandas as pd
import os

def clean_and_merge_data(movies, ratings):
    """
    Cleans and merges the movies and ratings datasets.
    """
    print("Starting data preprocessing...")
    
    # Drop timestamp from ratings as we won't use it for basic collaborative filtering
    if 'timestamp' in ratings.columns:
        ratings = ratings.drop('timestamp', axis=1)
        
    # Merge movies and ratings
    movie_ratings = pd.merge(ratings, movies, on='movieId')
    
    # Check for missing values
    missing_values = movie_ratings.isnull().sum()
    if missing_values.any():
        print("Missing values found and will be dropped:\n", missing_values)
        movie_ratings = movie_ratings.dropna()
        
    print(f"Merged dataset shape: {movie_ratings.shape}")
    return movie_ratings

def filter_sparse_data(movie_ratings, min_movie_ratings=20, min_user_ratings=20):
    """
    Filters out users who have rated few movies and movies that have few ratings
    to reduce sparsity and improve recommendations.
    """
    print(f"Filtering dataset (min_movie_ratings={min_movie_ratings}, min_user_ratings={min_user_ratings})...")
    
    # Filter movies
    movie_counts = movie_ratings['movieId'].value_counts()
    popular_movies = movie_counts[movie_counts >= min_movie_ratings].index
    filtered_ratings = movie_ratings[movie_ratings['movieId'].isin(popular_movies)]
    
    # Filter users
    user_counts = filtered_ratings['userId'].value_counts()
    active_users = user_counts[user_counts >= min_user_ratings].index
    filtered_ratings = filtered_ratings[filtered_ratings['userId'].isin(active_users)]
    
    print(f"Filtered dataset shape: {filtered_ratings.shape}")
    return filtered_ratings

def save_processed_data(df, output_path="data/processed/movie_ratings_processed.csv"):
    """
    Saves the preprocessed dataset.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    from data_loader import load_raw_data
    
    # Determine correct paths based on current working directory
    is_src = os.path.basename(os.getcwd()) == "src"
    raw_path = "../data/raw/ml-latest-small" if is_src else "data/raw/ml-latest-small"
    processed_path = "../data/processed/movie_ratings_processed.csv" if is_src else "data/processed/movie_ratings_processed.csv"
    
    movies, ratings = load_raw_data(raw_path)
    merged_data = clean_and_merge_data(movies, ratings)
    filtered_data = filter_sparse_data(merged_data)
    save_processed_data(filtered_data, processed_path)
