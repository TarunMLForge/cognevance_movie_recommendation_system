import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def load_data(filepath="data/processed/movie_ratings_processed.csv"):
    """Loads the processed movie dataset."""
    return pd.read_csv(filepath)

def plot_rating_distribution(df, output_dir="reports/figures"):
    """Plots and saves the distribution of ratings."""
    plt.figure(figsize=(8, 5))
    sns.countplot(x='rating', data=df, palette='viridis', hue='rating', legend=False)
    plt.title('Distribution of Movie Ratings')
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.savefig(os.path.join(output_dir, 'rating_distribution.png'))
    plt.close()

def plot_top_movies(df, output_dir="reports/figures", top_n=10):
    """Plots and saves the most frequently rated movies."""
    top_movies = df.groupby('title').size().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_movies.values, y=top_movies.index, palette='mako', hue=top_movies.index, legend=False)
    plt.title(f'Top {top_n} Most Rated Movies')
    plt.xlabel('Number of Ratings')
    plt.ylabel('Movie Title')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_movies.png'))
    plt.close()

def plot_genre_distribution(df, output_dir="reports/figures"):
    """Extracts, plots, and saves the distribution of movie genres."""
    # Split genres string by '|' and count occurrences
    all_genres = df['genres'].str.split('|', expand=True).stack().value_counts()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=all_genres.values, y=all_genres.index, palette='magma', hue=all_genres.index, legend=False)
    plt.title('Distribution of Movie Genres')
    plt.xlabel('Total Rating Count for Genre')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'genre_distribution.png'))
    plt.close()

if __name__ == "__main__":
    # Determine correct paths based on current working directory
    is_src = os.path.basename(os.getcwd()) == "src"
    data_path = "../data/processed/movie_ratings_processed.csv" if is_src else "data/processed/movie_ratings_processed.csv"
    output_dir = "../reports/figures" if is_src else "reports/figures"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading processed data...")
    df = load_data(data_path)
    
    print("Generating rating distribution plot...")
    plot_rating_distribution(df, output_dir)
    
    print("Generating top movies plot...")
    plot_top_movies(df, output_dir)
    
    print("Generating genre distribution plot...")
    plot_genre_distribution(df, output_dir)
    
    print(f"EDA complete! Visualizations saved to {output_dir}")
