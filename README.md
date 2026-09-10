# Movie Recommendation System 🎬

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Collaborative%20Filtering-orange)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-yellow)

## 📌 Overview
This repository contains an AI-based Movie Recommendation System built for Cognevance Technologies. It utilizes Item-Item Collaborative Filtering and Cosine Similarity to analyze historical user ratings and suggest relevant movies they are likely to enjoy.

## 🎯 Problem Statement
Users face choice overload on streaming platforms. This project aims to solve this by building a highly scalable and accurate recommendation engine that discovers hidden patterns in user preferences and suggests tailored movie choices.

## 📊 Dataset
The model is trained on the [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/) (100k ratings, 9k movies, 600 users). 
The data is preprocessed to remove sparsity (users with <20 ratings and movies with <20 ratings are removed) to improve the signal-to-noise ratio in the similarity matrix.

## 🛠 Technologies
- **Python** (Core language)
- **Pandas / NumPy** (Data manipulation and matrix operations)
- **Scikit-Learn** (Cosine Similarity calculation and RMSE evaluation)
- **Matplotlib / Seaborn** (Exploratory Data Analysis)

## 🚀 Project Workflow
1. **Data Loading & Preprocessing:** Merges raw datasets, handles missing values, and drops sparse entries.
2. **Exploratory Data Analysis (EDA):** Generates insightful visualizations about rating distributions and popular genres.
3. **Model Training:** Constructs a User-Item rating matrix and computes pairwise Cosine Similarity between items.
4. **Evaluation:** Evaluates model performance on a 20% holdout test set using RMSE.
5. **Interactive CLI:** Provides an interface for users to query movie recommendations dynamically.

## 📂 Folder Structure
```text
cognevance_movie_recommendation_system/
│
├── data/
│   ├── raw/               # Downloaded MovieLens CSV files
│   └── processed/         # Cleaned and filtered dataset
├── models/                # Serialized .pkl similarity matrix model
├── src/                   
│   ├── data_loader.py     # Script to load raw data safely
│   ├── preprocess.py      # Script to clean and merge datasets
│   ├── eda.py             # Script to generate visualizations
│   ├── recommender.py     # Script to train and save the CF model
│   ├── evaluate.py        # Script to evaluate RMSE on test data
│   └── cli_app.py         # Interactive command-line interface
├── reports/
│   ├── project_report.md  # Detailed project write-up
│   └── figures/           # Generated EDA visualizations
├── requirements.txt       # Project dependencies
└── README.md              # You are here
```

## ⚙️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cognevance_movie_recommendation_system.git
   cd cognevance_movie_recommendation_system
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. *(Optional if not already included)* Download the [MovieLens dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip) and extract it into `data/raw/ml-latest-small/`.

## 💻 Usage

To run the complete pipeline from scratch:

**1. Preprocess the Data:**
```bash
python src/preprocess.py
```

**2. Generate Exploratory Visualizations:**
```bash
python src/eda.py
```

**3. Train the Recommendation Model:**
```bash
python src/recommender.py
```

**4. Evaluate the Model (RMSE):**
```bash
python src/evaluate.py
```

**5. Launch the Interactive Recommendation Interface:**
```bash
python src/cli_app.py
```

## 📈 Results
Our implementation of Item-Item Collaborative Filtering achieved an **RMSE of 0.9124** on the test dataset. This indicates that our model's predicted rating deviates from a user's actual rating by less than a single star on average. Detailed explanations can be found in `reports/project_report.md`.

## 🔮 Future Improvements
- **Hybrid Recommendations:** Combine Collaborative Filtering with Content-Based filtering (using movie tags and genres) to address the "Cold Start" problem.
- **Deep Learning:** Implement neural collaborative filtering or Matrix Factorization (SVD) for deeper feature extraction.
- **Web API:** Deploy the backend logic using FastAPI or Flask to serve recommendations via a RESTful API.

## 👨‍💻 Author
Developed for Cognevance Technologies Artificial Intelligence & Machine Learning Projects.
