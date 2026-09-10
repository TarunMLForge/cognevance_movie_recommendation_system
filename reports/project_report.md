# Project Report: Movie Recommendation System

## 1. Problem Statement
With the vast amount of content available on streaming platforms, users often experience choice overload. A recommendation system solves this by analyzing user behavior and suggesting relevant movies they are likely to enjoy. The objective of this project is to build an AI-based recommendation engine that filters and suggests movies based on collaborative interactions.

## 2. Dataset
This project uses the **MovieLens Small Dataset** collected by GroupLens Research.
- **Movies:** ~9,000 unique movies across multiple genres.
- **Ratings:** ~100,000 ratings provided by 600 distinct users on a scale of 0.5 to 5.0.

## 3. Data Preprocessing
To prepare the data for our machine learning model, we performed the following preprocessing steps:
- Merged `movies.csv` and `ratings.csv` on the `movieId` attribute.
- Removed irrelevant columns like `timestamp`.
- **Sparsity Reduction:** Collaborative filtering models suffer when there is not enough intersecting data (sparsity). To mitigate this, we dropped users who rated fewer than 20 movies and movies that received fewer than 20 ratings. This reduced the dataset to highly meaningful interactions (~67k ratings).

## 4. Recommendation Approach
We utilized **Item-Item Collaborative Filtering**.
Unlike User-User CF which finds similar users, Item-Item CF finds similar movies based on how users have rated them. If a large group of users rated both *Movie A* and *Movie B* highly, the items are considered similar.

## 5. Algorithm (Cosine Similarity)
The core algorithm relies on computing the **Cosine Similarity** between movie vectors.
We transformed the dataset into a User-Item sparse matrix where rows represent users, columns represent movies, and cell values represent the ratings (with unrated movies filled as 0). The cosine similarity formula measures the cosine of the angle between two multi-dimensional vectors (movies), generating a similarity score between 0 and 1. 

## 6. Results & Evaluation
The model successfully identifies non-trivial relationships between movies.
To quantitatively evaluate performance, we partitioned 20% of the ratings as a test set. We built the model on the remaining 80% and predicted the missing user ratings on the test set.
- **Metric:** Root Mean Squared Error (RMSE)
- **Result:** ~0.9124 

An RMSE of 0.91 means our predicted ratings deviate from actual ratings by less than 1 star on average, which is highly competitive for a standard CF implementation.

## 7. Visualizations
During our Exploratory Data Analysis, we identified key trends:
- **Rating Distribution:** Users predominantly give positive scores, with `4.0` and `3.0` being the most common.
- **Top Movies:** Blockbusters like *Forrest Gump*, *The Shawshank Redemption*, and *Pulp Fiction* have the highest volume of ratings.
- **Genres:** Drama and Comedy are the most prolific genres in the dataset, followed closely by Action and Thriller.

*(Visualizations are available in the `reports/figures/` directory).*

## 8. Limitations
- **Cold Start Problem:** The model cannot recommend movies to a brand-new user with zero ratings, nor can it recommend a brand-new movie with zero ratings.
- **Popularity Bias:** The model tends to strongly recommend highly popular blockbusters over niche hidden gems.

## 9. Future Improvements
- Implement a hybrid recommendation engine combining Collaborative Filtering with Content-Based Filtering (using movie genres/tags) to solve the cold start problem.
- Implement Matrix Factorization (e.g., SVD) for potentially better scaling and dimensionality reduction.
- Deploy the system using a web framework (like FastAPI or Flask) to allow remote access.
