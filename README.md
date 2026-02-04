# Insurance Charges Prediction - Complete ML Pipeline

## 📋 Overview
Complete machine learning pipeline for predicting insurance charges with:
- Automated data preprocessing and feature engineering
- Training and comparison of 13 different ML algorithms
- Model evaluation with multiple metrics
- Best model selection and deployment
- Interactive Streamlit web app for predictions

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run ML Pipeline
```bash
python ml_pipeline.py
```

This will:
- Load and preprocess data
- Engineer features
- Train 13 different models
- Compare performance metrics
- Select best model
- Save model as `insurance_model.pkl`
- Generate evaluation report and visualizations

### 3. Launch Web App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Project Structure
```
├── insurance_data.csv              # Input dataset
├── ml_pipeline.py                  # ML pipeline script
├── app.py                          # Streamlit web app
├── requirements.txt                # Python dependencies
├── insurance_model.pkl             # Trained model (generated)
├── model_evaluation_results.csv    # Performance metrics (generated)
└── model_comparison.png            # Visualization (generated)
```

## 🤖 Models Trained
1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. Decision Tree
5. Random Forest
6. Gradient Boosting
7. XGBoost
8. LightGBM
9. Extra Trees
10. AdaBoost
11. K-Nearest Neighbors
12. Support Vector Regression

## 📊 Evaluation Metrics
- **R² Score:** Model accuracy
- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **MAPE:** Mean Absolute Percentage Error
- **Cross-Validation:** 5-fold CV for robustness

## ✨ Features
- Automated feature engineering (10+ new features)
- Interaction features (smoker × BMI, age × BMI)
- Risk scoring
- BMI and age categorization
- Model comparison visualization
- Best model selection based on multiple criteria

## 🎯 Web App Features
- User-friendly interface
- Real-time predictions
- Risk assessment
- Cost breakdown analysis
- BMI calculator reference
- Health recommendations

## 📈 Expected Performance
- R² Score: ~0.85-0.90
- MAE: ~$2,000-$2,500
- Models typically rank: XGBoost > LightGBM > Gradient Boosting > Random Forest


