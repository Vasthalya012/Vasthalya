import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess_data(filepath='dataset/admission_dataset.csv'):
    # Load dataset
    df = pd.read_csv(filepath)
    
    # 1. Handle missing values
    # Fill numerical NA with median
    for col in ['GRE Score', 'TOEFL Score', 'CGPA']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
            
    # 2. Remove duplicates
    df = df.drop_duplicates()
    
    # 3. Handle outliers (Optional, basic capping)
    # E.g., for entrance exam score, we can cap based on quantiles
    # For simplicity in this project, we'll keep it standard unless extremes exist
    
    # Separate features and target
    X = df.drop('Admission Status', axis=1)
    y = df['Admission Status']
    
    # 4. Encoding
    # Target encoding (Admitted -> 1, Not Admitted -> 0)
    le_y = LabelEncoder()
    y = le_y.fit_transform(y)
    
    # Categorical features encoding (One-Hot or Label)
    cat_cols = ['Category', 'Gender', 'Preferred Course', 'Preferred University']
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)
    
    feature_names = X_encoded.columns.tolist()
    
    # 5. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42, stratify=y)
    
    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le_y, feature_names, df

def preprocess_input(input_dict, scaler, feature_names):
    """
    Takes a dictionary of inputs from Streamlit, scales/encodes it, 
    and returns a numpy array ready for prediction.
    """
    df_input = pd.DataFrame([input_dict])
    
    cat_cols = ['Category', 'Gender', 'Preferred Course', 'Preferred University']
    df_input_encoded = pd.get_dummies(df_input, columns=cat_cols)
    
    # Ensure all columns exist as in training data
    for col in feature_names:
        if col not in df_input_encoded.columns:
            df_input_encoded[col] = 0
            
    # Reorder columns to match training
    df_input_encoded = df_input_encoded[feature_names]
    
    # Scale
    input_scaled = scaler.transform(df_input_encoded)
    return input_scaled

