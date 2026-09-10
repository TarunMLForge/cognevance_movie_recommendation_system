import streamlit as st
import pandas as pd
import os
import sys

# Add src to path so we can import recommender
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
try:
    from recommender import load_model, get_movie_recommendations
except ModuleNotFoundError:
    st.error("Could not find the recommender module. Make sure you are running from the correct directory.")

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 AI Movie Recommendation System")
st.markdown("Select a movie you love, and our AI will recommend 5 similar movies based on collaborative filtering!")

# Load model using caching so it doesn't reload on every interaction
@st.cache_data
def load_similarity_model():
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'item_similarity_model.pkl')
    if not os.path.exists(model_path):
        return None
    return load_model(model_path)

similarity_df = load_similarity_model()

if similarity_df is None:
    st.warning("⚠️ Model not found! Please run `python src/recommender.py` in your terminal to train and generate the model first.")
else:
    movies_list = similarity_df.columns.tolist()
    
    # Dropdown for movie selection
    selected_movie = st.selectbox(
        "Search and select a movie you like:",
        movies_list,
        index=movies_list.index("Matrix, The (1999)") if "Matrix, The (1999)" in movies_list else 0
    )
    
    if st.button("Generate Recommendations 🚀"):
        with st.spinner('Finding the best matches...'):
            recommendations = get_movie_recommendations(selected_movie, similarity_df, top_n=5)
            
            st.success("Recommendations generated!")
            st.subheader(f"Since you liked **{selected_movie}**, we recommend:")
            
            # Display recommendations elegantly
            for i, (title, score) in enumerate(recommendations.items(), 1):
                st.markdown(f"**{i}. {title}** *(Similarity Score: {score:.2f})*")
