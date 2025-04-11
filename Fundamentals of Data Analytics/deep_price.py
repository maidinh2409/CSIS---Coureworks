# ==============================================
# Rental Price Classification Project - FIXED VERSION
# ==============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, 
                            confusion_matrix, 
                            accuracy_score,
                            f1_score)
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# ==============================================
# Phase 1: Discovery - Data Loading
# ==============================================

# Load your dataset
try:
    df = pd.read_csv(r'C:\Users\ridor\OneDrive - Douglas College\T04_Winter2025\CSIS-3360-001_Fund-DA\group_project\clasification\rental_data.csv')
    print("Data loaded successfully. Shape:", df.shape)
except Exception as e:
    print("Error loading data:", e)
    exit()

# Clean column names by removing spaces and special characters
df.columns = df.columns.str.replace(' ', '_').str.replace('/', '_')
print("\nCleaned column names:", df.columns.tolist())

# ==============================================
# Phase 2: Data Preparation
# ==============================================

print("\n=== Data Cleaning ===")

# Handle missing values if any
print("\nMissing values before cleaning:")
print(df.isnull().sum())

# Fill or drop missing values (adjust based on your data)
df = df.dropna()  # or use df.fillna(...)

# Convert categorical columns - using cleaned column names
df['Pets_Yes_NO'] = df['Pets_Yes_NO'].map({'Yes': 1, 'No': 0})
df['fee'] = df['fee'].map({'No': 0, 'Yes': 1})

# Create price categories (Low, Mid-range, High)
df['price_category'] = pd.qcut(df['price'], 
                             q=[0, 0.25, 0.75, 1], 
                             labels=['Low', 'Mid-range', 'High'])

# Feature engineering
df['price_per_sqft'] = df['price'] / df['square_feet']

# ==============================================
# Phase 3: Model Planning (EDA)
# ==============================================

print("\n=== Exploratory Data Analysis ===")

# Numerical features distribution
num_features = ['bathrooms', 'bedrooms', 'square_feet', 'Count_of_Amenities', 'price_per_sqft']
df[num_features].hist(bins=30, figsize=(15, 10))
plt.suptitle('Numerical Features Distribution')
plt.tight_layout()
plt.show()

# ==============================================
# Phase 4: Model Building
# ==============================================

# Define features and target
X = df.drop(['price', 'price_category', 'cityname', 'state'], axis=1)
y = df['price_category']

# Identify feature types
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print("\nNumeric features:", numeric_features)
print("Categorical features:", categorical_features)

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Random Forest Pipeline
print("\nTraining Random Forest...")
rf_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    ))
])

rf_pipeline.fit(X_train, y_train)

# ==============================================
# Model Evaluation
# ==============================================

def evaluate_model(model, X_test, y_test, model_name):
    print(f"\n=== {model_name} Evaluation ===")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=model.classes_, 
                yticklabels=model.classes_)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

evaluate_model(rf_pipeline, X_test, y_test, "Random Forest")

# ==============================================
# Feature Importance
# ==============================================

# Get feature names after preprocessing
preprocessor.fit(X)
feature_names = numeric_features + \
               list(rf_pipeline.named_steps['preprocessor']
                   .named_transformers_['cat']
                   .get_feature_names_out(categorical_features))

# Get feature importances
importances = rf_pipeline.named_steps['classifier'].feature_importances_

# Create importance DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=importance_df.head(10), x='Importance', y='Feature')
plt.title('Top 10 Feature Importances (Random Forest)')
plt.tight_layout()
plt.show()

# Feature Importance Plot (simplified)
top_features = importance_df.head(5)
plt.figure(figsize=(10,5))
sns.barplot(x='Importance', y='Feature', data=top_features, palette='Blues_d')
plt.title('What Drives Rental Prices?')
plt.xlabel('Impact on Price Classification')
plt.tight_layout()

# Save the model
import joblib
joblib.dump(rf_pipeline, 'rental_price_classifier.pkl')
print("\nModel saved as 'rental_price_classifier.pkl'")