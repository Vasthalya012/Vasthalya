import joblib
import os
from preprocessing import preprocess_input

def load_artifacts():
    try:
        model = joblib.load('models/best_model.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        le_y = joblib.load('models/label_encoder.joblib')
        return model, scaler, feature_names, le_y
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None, None, None

def predict_admission(input_dict):
    model, scaler, feature_names, le_y = load_artifacts()
    
    if model is None:
        return "Error", 0, "Models not trained. Please run training script first."
        
    input_scaled = preprocess_input(input_dict, scaler, feature_names)
    
    # Predict
    prediction = model.predict(input_scaled)
    prediction_label = le_y.inverse_transform(prediction)[0]
    
    # Probability
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = probabilities[prediction[0]] * 100
    else:
        confidence = 100.0 # fallback
        
    # Recommendation logic
    if prediction_label == 'Admitted':
        if confidence >= 80:
            rec = "Highly Eligible: You have a very strong profile. Focus on finalizing your SOP."
        elif confidence >= 60:
            rec = "Good Chance: Your profile is competitive. Ensure strong LORs and extracurriculars."
        else:
            rec = "Moderate Chance: Consider improving test scores or gaining more research experience."
    else:
        if confidence >= 70:
            rec = "Low Chance: Profile is weak. Significant improvements needed in entrance exams and CGPA."
        else:
            rec = "Moderate Chance (Not Admitted): Borderline. Work on enhancing your SOP and gaining work experience."
            
    return prediction_label, confidence, rec
