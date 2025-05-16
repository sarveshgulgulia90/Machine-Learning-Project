# 🎬 Movie Success Prediction using Ensemble Learning

## 📌 Project Overview
This project focuses on predicting the success of movies using multiple machine learning models. The goal is to classify movies based on their Return on Investment (ROI) into categories such as "Hit", "Average", and "Flop". We used an ensemble approach to improve performance.

## 🗃️ Dataset
- Source: [[TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies))
- Features include budget, revenue, genres, popularity, and more.
- Preprocessing involved ROI calculation and genre one-hot encoding.

## ⚙️ Features Used
- `budget`
- `revenue`
- `genres` (one-hot encoded)
- `popularity`
- `vote_average`
- `vote_count`
- `release_year`

> Note: `original_language` and `runtime` were excluded from the KNN model only.

## 🧠 Models Used
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree Classifier
- Random Forest Classifier
- Neural Network (MLP)
- Ensemble Voting (Bagging + Majority Voting)

## 📊 Results
- Evaluation Metric: Weighted F1-Score
- Ensemble model achieved the highest F1-Score across all models
- Action and Sci-Fi genres were most correlated with higher ROI

Run the notebook or Python scripts to train and evaluate models.
Future Work
Hyperparameter tuning for all models
Additional features like cast, director popularity
Deep learning architectures (e.g., LSTM for temporal data)
✍️ Contributors
Sarvesh Gulgulia(me)
 Teammate’s Name
 Gaurav Singh Bora
 Rishi Sharma
 Arshveer Singh Chauhan
