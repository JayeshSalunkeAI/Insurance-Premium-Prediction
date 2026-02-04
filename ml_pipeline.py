# ============================================================================
# ML PIPELINE - Insurance Charges Prediction
# ============================================================================

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, 
                             mean_absolute_percentage_error)

# Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor, 
                              AdaBoostRegressor, ExtraTreesRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

print("="*80)
print("STEP 1: DATA LOADING & EXPLORATION")
print("="*80)

# Load data
df = pd.read_csv('insurance_data.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nMissing values: {df.isnull().sum().sum()}")

# ============================================================================
print("\n" + "="*80)
print("STEP 2: FEATURE ENGINEERING")
print("="*80)

df_processed = df.copy()

# Encode categorical variables
le_sex = LabelEncoder()
le_smoker = LabelEncoder()
le_region = LabelEncoder()

df_processed['sex_encoded'] = le_sex.fit_transform(df_processed['sex'])
df_processed['smoker_encoded'] = le_smoker.fit_transform(df_processed['smoker'])
df_processed['region_encoded'] = le_region.fit_transform(df_processed['region'])
df_processed['smoker_binary'] = (df_processed['smoker'] == 'yes').astype(int)

# Create engineered features
print("Creating engineered features...")

# BMI categories
def categorize_bmi(bmi):
    if bmi < 18.5:
        return 0
    elif bmi < 25:
        return 1
    elif bmi < 30:
        return 2
    else:
        return 3

df_processed['bmi_category'] = df_processed['bmi'].apply(categorize_bmi)

# Age groups
df_processed['age_group'] = pd.cut(df_processed['age'], 
                                    bins=[0, 30, 50, 100], 
                                    labels=[0, 1, 2]).astype(int)

# Interaction features
df_processed['smoker_bmi'] = df_processed['smoker_binary'] * df_processed['bmi']
df_processed['age_bmi'] = df_processed['age'] * df_processed['bmi']
df_processed['is_obese_smoker'] = ((df_processed['bmi'] > 30) & 
                                    (df_processed['smoker_binary'] == 1)).astype(int)

# Risk score
df_processed['risk_score'] = (
    df_processed['age'] / 64 * 0.3 + 
    df_processed['bmi'] / 54 * 0.3 + 
    df_processed['smoker_binary'] * 0.4
)

print(f"Total features: {df_processed.shape[1]}")

# ============================================================================
print("\n" + "="*80)
print("STEP 3: PREPARE DATA FOR MODELING")
print("="*80)

# Select features
feature_columns = ['age', 'bmi', 'children', 'smoker_encoded', 
                   'bmi_category', 'age_group', 'smoker_bmi', 
                   'age_bmi', 'is_obese_smoker', 'risk_score']

X = df_processed[feature_columns]
y = df_processed['charges']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scaler for specific models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Features: {X_train.shape[1]}")

# ============================================================================
print("\n" + "="*80)
print("STEP 4: TRAIN MULTIPLE MODELS")
print("="*80)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=1.0),
    'Decision Tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                          random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5,
                                                   random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, 
                           learning_rate=0.1, random_state=42, verbosity=0),
    'LightGBM': LGBMRegressor(n_estimators=100, max_depth=5, 
                             learning_rate=0.1, random_state=42, verbose=-1),
    'Extra Trees': ExtraTreesRegressor(n_estimators=100, max_depth=10, 
                                       random_state=42, n_jobs=-1),
    'AdaBoost': AdaBoostRegressor(n_estimators=100, random_state=42),
    'KNN': KNeighborsRegressor(n_neighbors=5),
    'SVR': SVR(kernel='rbf', C=1000, gamma=0.1)
}

results = []

print("\nTraining models...\n")

for name, model in models.items():
    print(f"Training {name}...")

    # Use scaled data for specific models
    if name in ['KNN', 'SVR', 'Ridge Regression', 'Lasso Regression']:
        X_train_model = X_train_scaled
        X_test_model = X_test_scaled
    else:
        X_train_model = X_train
        X_test_model = X_test

    # Train
    model.fit(X_train_model, y_train)

    # Predictions
    y_train_pred = model.predict(X_train_model)
    y_test_pred = model.predict(X_test_model)

    # Metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred) * 100

    # Cross-validation
    if name in ['KNN', 'SVR', 'Ridge Regression', 'Lasso Regression']:
        cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)
    else:
        cv_scores = cross_val_score(model, X_train, y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)

    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results.append({
        'Model': name,
        'Train_R2': train_r2,
        'Test_R2': test_r2,
        'CV_R2_Mean': cv_mean,
        'CV_R2_Std': cv_std,
        'MAE': test_mae,
        'RMSE': test_rmse,
        'MAPE_%': test_mape,
        'Overfit': train_r2 - test_r2
    })

    print(f"  ✓ R²: {test_r2:.4f} | MAE: ${test_mae:,.2f} | CV: {cv_mean:.4f}±{cv_std:.3f}")

# ============================================================================
print("\n" + "="*80)
print("STEP 5: MODEL EVALUATION & COMPARISON")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Test_R2', ascending=False)

print("\nMODEL PERFORMANCE RANKING:")
print("="*80)
print(results_df[['Model', 'Test_R2', 'MAE', 'RMSE', 'CV_R2_Mean']].to_string(index=False))

# Save results
results_df.to_csv('model_evaluation_results.csv', index=False)
print("\n✓ Results saved to 'model_evaluation_results.csv'")

# ============================================================================
print("\n" + "="*80)
print("STEP 6: SELECT BEST MODEL")
print("="*80)

best_model_name = results_df.iloc[0]['Model']
best_test_r2 = results_df.iloc[0]['Test_R2']
best_mae = results_df.iloc[0]['MAE']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   Test R²: {best_test_r2:.4f}")
print(f"   MAE: ${best_mae:,.2f}")
print(f"   RMSE: ${results_df.iloc[0]['RMSE']:,.2f}")
print(f"   CV R²: {results_df.iloc[0]['CV_R2_Mean']:.4f} ± {results_df.iloc[0]['CV_R2_Std']:.3f}")

# Retrain best model on all data
best_model = models[best_model_name]

if best_model_name in ['KNN', 'SVR', 'Ridge Regression', 'Lasso Regression']:
    X_final = scaler.fit_transform(X)
    best_model.fit(X_final, y)
    use_scaling = True
else:
    best_model.fit(X, y)
    use_scaling = False

# ============================================================================
print("\n" + "="*80)
print("STEP 7: SAVE MODEL PACKAGE")
print("="*80)

model_package = {
    'model': best_model,
    'model_name': best_model_name,
    'scaler': scaler if use_scaling else None,
    'feature_names': list(X.columns),
    'use_scaling': use_scaling,
    'label_encoders': {
        'sex': le_sex,
        'smoker': le_smoker,
        'region': le_region
    },
    'performance': {
        'test_r2': float(best_test_r2),
        'mae': float(best_mae),
        'rmse': float(results_df.iloc[0]['RMSE'])
    }
}

with open('insurance_model.pkl', 'wb') as f:
    pickle.dump(model_package, f)

print("\n✓ Model saved as 'insurance_model.pkl'")
print("\nPackage contains:")
print(f"  - Model: {best_model_name}")
print(f"  - Features: {len(model_package['feature_names'])}")
print(f"  - Scaler: {'Yes' if use_scaling else 'No'}")
print(f"  - Encoders: sex, smoker, region")

# ============================================================================
print("\n" + "="*80)
print("STEP 8: VISUALIZE RESULTS")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# R² Score comparison
ax1 = axes[0, 0]
y_pos = np.arange(len(results_df))
colors = ['green' if i == 0 else 'lightblue' for i in range(len(results_df))]
ax1.barh(y_pos, results_df['Test_R2'], color=colors, alpha=0.7, edgecolor='black')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(results_df['Model'], fontsize=9)
ax1.set_xlabel('R² Score', fontweight='bold')
ax1.set_title('Model Accuracy (R²)', fontweight='bold', fontsize=12)
ax1.grid(axis='x', alpha=0.3)

# MAE comparison
ax2 = axes[0, 1]
ax2.barh(y_pos, results_df['MAE'], color=colors, alpha=0.7, edgecolor='black')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(results_df['Model'], fontsize=9)
ax2.set_xlabel('MAE ($)', fontweight='bold')
ax2.set_title('Prediction Error', fontweight='bold', fontsize=12)
ax2.grid(axis='x', alpha=0.3)

# Train vs Test R²
ax3 = axes[1, 0]
x = np.arange(len(results_df))
width = 0.35
ax3.bar(x - width/2, results_df['Train_R2'], width, label='Train', alpha=0.7)
ax3.bar(x + width/2, results_df['Test_R2'], width, label='Test', alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax3.set_ylabel('R²', fontweight='bold')
ax3.set_title('Overfitting Check', fontweight='bold', fontsize=12)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

# Cross-Validation
ax4 = axes[1, 1]
ax4.errorbar(range(len(results_df)), results_df['CV_R2_Mean'], 
             yerr=results_df['CV_R2_Std'], fmt='o', markersize=8, 
             capsize=5, capthick=2, alpha=0.7)
ax4.set_xticks(range(len(results_df)))
ax4.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=8)
ax4.set_ylabel('CV R²', fontweight='bold')
ax4.set_title('Cross-Validation (5-Fold)', fontweight='bold', fontsize=12)
ax4.grid(alpha=0.3)

plt.suptitle('ML Model Comparison - Insurance Charges', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualization saved as 'model_comparison.png'")

print("\n" + "="*80)
print("PIPELINE COMPLETED!")
print("="*80)
print("\nGenerated files:")
print("  1. insurance_model.pkl")
print("  2. model_evaluation_results.csv")
print("  3. model_comparison.png")
print("\nNext: Run 'streamlit run app.py'")
