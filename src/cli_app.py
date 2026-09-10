import pandas as pd
import os
import sys

from src.recommender import load_model, get_movie_recommendations

def main():
    print("==================================================")
    print("       MOVIE RECOMMENDATION SYSTEM CLI")
    print("==================================================")
    
    is_src = os.path.basename(os.getcwd()) == "src"
    model_path = "../models/item_similarity_model.pkl" if is_src else "models/item_similarity_model.pkl"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        print("Please run 'python src/recommender.py' first to generate the model.")
        sys.exit(1)
        
    print("Loading recommendation model (this might take a few seconds)...")
    item_similarity_df = load_model(model_path)
    
    print("\nModel loaded successfully!")
    print("Type 'exit' or 'quit' at any time to stop.")
    
    while True:
        print("\n--------------------------------------------------")
        movie_title = input("Enter a movie title you like (e.g., 'Toy Story (1995)'): ").strip()
        
        if movie_title.lower() in ['exit', 'quit']:
            print("Exiting Movie Recommendation System. Goodbye!")
            break
            
        if not movie_title:
            continue
            
        recommendations = get_movie_recommendations(movie_title, item_similarity_df, top_n=5)
        
        # get_movie_recommendations returns a string if the movie isn't found
        if isinstance(recommendations, str):
            print(f"\n{recommendations}")
            print("Tip: Ensure you include the release year in parentheses exactly as formatted in the dataset (e.g., 'Inception (2010)').")
        else:
            print(f"\nSince you liked '{movie_title}', we recommend:")
            for i, (title, score) in enumerate(recommendations.items(), 1):
                print(f"  {i}. {title} (Similarity Score: {score:.3f})")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting Movie Recommendation System. Goodbye!")
