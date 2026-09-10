import pandas as pd
import os

def load_raw_data(data_dir="data/raw/ml-latest-small"):
    """
    Loads movies and ratings data from the raw directory.
    """
    movies_path = os.path.join(data_dir, "movies.csv")
    ratings_path = os.path.join(data_dir, "ratings.csv")
    
    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Dataset not found at {data_dir}. Please ensure the MovieLens dataset is extracted there.")
        
    print(f"Loading movies from: {movies_path}")
    movies = pd.read_csv(movies_path)
    
    print(f"Loading ratings from: {ratings_path}")
    ratings = pd.read_csv(ratings_path)
    
    return movies, ratings

if __name__ == "__main__":
    # For testing the script directly
    import sys
    # Adjust path if run from inside src/
    data_path = "../data/raw/ml-latest-small" if os.path.basename(os.getcwd()) == "src" else "data/raw/ml-latest-small"
    movies, ratings = load_raw_data(data_path)
    print("Movies shape:", movies.shape)
    print("Ratings shape:", ratings.shape)
