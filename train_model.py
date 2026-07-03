import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from preprocessing import load_and_preprocess_data

def train_and_evaluate():
    print("Loading and preprocessing data...")
    X_train, X_test, y_train, y_test, scaler, le_y, feature_names, df = load_and_preprocess_data()
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'KNN': KNeighborsClassifier(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }
    
    results = []
    best_model = None
    best_roc_auc = -1
    best_model_name = ""
    
    print("Training models...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else [0]*len(y_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC AUC only if probabilities are available
        try:
            roc_auc = roc_auc_score(y_test, y_proba)
        except:
            roc_auc = 0
            
        results.append({
            'Model': name,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        })
        
        print(f"{name} -> Accuracy: {acc:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model = model
            best_model_name = name
            
    print(f"\nBest Model: {best_model_name} with ROC-AUC: {best_roc_auc:.4f}")
    
    os.makedirs('models', exist_ok=True)
    
    # Save best model, scaler, and feature names
    joblib.dump(best_model, f'models/best_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(feature_names, 'models/feature_names.joblib')
    joblib.dump(le_y, 'models/label_encoder.joblib')
    
    print("Model artifacts saved in models/ directory.")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('models/model_performance.csv', index=False)

if __name__ == "__main__":
    train_and_evaluate()
