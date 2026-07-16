# 🏠 House Price Predictor

A machine learning web app that predicts house prices based on property features like area, bedrooms, bathrooms, age, distance from city, and locality demand.

## Tech Stack
- Python, scikit-learn (Random Forest Regression)
- Pandas for data processing
- Streamlit for the web interface

## How it works
1. `train_model.py` generates a synthetic housing dataset and trains a Random Forest Regressor (R² ≈ 0.88 on test data).
2. `app.py` loads the trained model and serves an interactive Streamlit UI where users adjust sliders and get an instant price estimate.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py   # trains and saves the model
streamlit run app.py    # launches the web app
```

## Live Demo
🔗 [Add your deployed Streamlit link here]

## Author
Sowmiga S V
