import pandas as pd
import numpy as np
import os

def generate_synthetic_dataset(num_samples=2000):
    np.random.seed(42)
    
    # 1. GRE Score (290 to 340)
    gre_scores = np.random.randint(290, 341, num_samples)
    
    # 2. TOEFL Score (90 to 120)
    toefl_scores = np.random.randint(90, 121, num_samples)
    
    # 3. University Rating (1 to 5)
    univ_rating = np.random.randint(1, 6, num_samples)
    
    # 4. SOP Strength (1.0 to 5.0)
    sop = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], num_samples)
    
    # 5. LOR Strength (1.0 to 5.0)
    lor = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0], num_samples)
    
    # 6. CGPA (6.8 to 10.0)
    cgpa = np.round(np.random.uniform(6.8, 10.0, num_samples), 2)
    
    # 7. Research Experience (0 or 1)
    research = np.random.randint(0, 2, num_samples)
    
    # 8. Work Experience (in months, 0 to 60)
    work_exp = np.random.randint(0, 61, num_samples)
    
    # 9. Extracurricular Activities (0 or 1)
    extra_curr = np.random.randint(0, 2, num_samples)
    
    # 10. Entrance Exam Score (percentage 50 to 100)
    entrance_score = np.round(np.random.uniform(50.0, 100.0, num_samples), 2)
    
    # 11. Category (General, OBC, SC, ST)
    categories = ['General', 'OBC', 'SC', 'ST']
    category = np.random.choice(categories, num_samples, p=[0.5, 0.25, 0.15, 0.10])
    
    # 12. Gender (Male, Female, Other)
    gender = np.random.choice(['Male', 'Female', 'Other'], num_samples, p=[0.55, 0.43, 0.02])
    
    # 13. Age (20 to 30)
    age = np.random.randint(20, 31, num_samples)
    
    # 14. Preferred Course
    courses = ['Computer Science', 'Data Science', 'Business Analytics', 'Electrical Engineering', 'Mechanical Engineering', 'Information Technology']
    preferred_course = np.random.choice(courses, num_samples)
    
    # 15. Preferred University
    universities = ['MIT', 'Stanford', 'Harvard', 'CMU', 'UC Berkeley', 'Oxford', 'Cambridge', 'ETH Zurich']
    preferred_university = np.random.choice(universities, num_samples)
    
    # Calculate Admission Chance (Base logic)
    # Higher scores -> Higher chance
    # Normalize some keys to create a probability
    prob = (gre_scores - 290) / 50 * 0.25 + \
           (toefl_scores - 90) / 30 * 0.15 + \
           (cgpa - 6.8) / 3.2 * 0.30 + \
           (univ_rating / 5) * 0.10 + \
           research * 0.05 + \
           (entrance_score - 50) / 50 * 0.10 + \
           (work_exp / 60) * 0.05

    # Add some random noise
    prob = prob + np.random.normal(0, 0.05, num_samples)
    
    # Admission Status (Admitted: 1, Not Admitted: 0)
    # Threshold for admission (approx 55% get admitted)
    admission_status = (prob >= 0.55).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'GRE Score': gre_scores,
        'TOEFL Score': toefl_scores,
        'University Rating': univ_rating,
        'SOP': sop,
        'LOR': lor,
        'CGPA': cgpa,
        'Research Experience': research,
        'Work Experience': work_exp,
        'Extracurricular Activities': extra_curr,
        'Entrance Exam Score': entrance_score,
        'Category': category,
        'Gender': gender,
        'Age': age,
        'Preferred Course': preferred_course,
        'Preferred University': preferred_university,
        'Admission Status': admission_status
    })
    
    # Map back to Admitted / Not Admitted for clarity
    df['Admission Status'] = df['Admission Status'].map({1: 'Admitted', 0: 'Not Admitted'})
    
    # Introduce some missing values to test preprocessing (e.g., 2% missing in some columns)
    cols_with_na = ['GRE Score', 'TOEFL Score', 'CGPA']
    for col in cols_with_na:
        idx = np.random.choice(df.index, size=int(num_samples * 0.02), replace=False)
        df.loc[idx, col] = np.nan
        
    os.makedirs('dataset', exist_ok=True)
    df.to_csv('dataset/admission_dataset.csv', index=False)
    print(f"Dataset generated with {num_samples} records at dataset/admission_dataset.csv")

if __name__ == "__main__":
    generate_synthetic_dataset()
