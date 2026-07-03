import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
from PIL import Image

from prediction import predict_admission
from utils import init_db, register_user, authenticate_user, save_prediction, get_user_history, get_all_history, generate_pdf_report

# Configure Page
st.set_page_config(page_title="AI Admission System", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

# Initialize Database
init_db()

# --- Custom CSS for Styling ---
def local_css():
    st.markdown("""
    <style>
        .main-header { font-size: 40px !important; font-weight: bold; color: #1E3A8A; text-align: center; }
        .sub-header { font-size: 24px !important; font-weight: bold; color: #2563EB; }
        .card { padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2); transition: 0.3s; margin-bottom: 20px; background-color: #f8fafc; }
        .card:hover { box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2); }
        .pred-admitted { color: #15803d; font-size: 32px; font-weight: bold; text-align: center; }
        .pred-rejected { color: #b91c1c; font-size: 32px; font-weight: bold; text-align: center; }
        .stButton>button { width: 100%; border-radius: 5px; font-weight: bold; background-color: #2563EB; color: white; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# --- Session State for Auth ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'is_admin' not in st.session_state:
    st.session_state.is_admin = False

# --- Navigation ---
def main():
    if not st.session_state.logged_in:
        auth_page()
    else:
        st.sidebar.title(f"Welcome, {st.session_state.username}")
        if st.sidebar.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ''
            st.session_state.is_admin = False
            st.rerun()
            
        pages = ["Home", "Admission Prediction", "Dataset Overview", "Model Performance", "My History", "About Project", "Contact"]
        if st.session_state.is_admin:
            pages.append("Admin Dashboard")
            
        choice = st.sidebar.radio("Navigate", pages)
        
        if choice == "Home":
            home_page()
        elif choice == "Admission Prediction":
            prediction_page()
        elif choice == "Dataset Overview":
            dataset_page()
        elif choice == "Model Performance":
            model_performance_page()
        elif choice == "My History":
            history_page()
        elif choice == "About Project":
            about_page()
        elif choice == "Contact":
            contact_page()
        elif choice == "Admin Dashboard":
            admin_page()

# --- Pages ---
def auth_page():
    st.markdown('<p class="main-header">🎓 AI-Enabled Student Admission System</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login")
            username = st.text_input("Username", key="login_user")
            password = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                if username and password:
                    auth, is_admin = authenticate_user(username, password)
                    if auth:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.is_admin = is_admin
                        st.success("Logged in successfully!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password.")
                else:
                    st.warning("Please fill all fields.")
                    
        with tab2:
            st.subheader("Register")
            new_user = st.text_input("New Username", key="reg_user")
            new_pass = st.text_input("New Password", type="password", key="reg_pass")
            if st.button("Register"):
                if new_user and new_pass:
                    if register_user(new_user, new_pass):
                        st.success("Registered successfully! Please login.")
                    else:
                        st.error("Username already exists.")
                else:
                    st.warning("Please fill all fields.")
        st.markdown('</div>', unsafe_allow_html=True)

def home_page():
    st.markdown('<p class="main-header">Welcome to AI Admission Predictor</p>', unsafe_allow_html=True)
    st.image("https://images.unsplash.com/photo-1523050854058-8df90110c9f1?ixlib=rb-1.2.1&auto=format&fit=crop&w=1950&q=80", use_container_width=True)
    
    st.markdown("""
    ### 🎯 Overview
    The AI-Enabled Student Admission System uses advanced Machine Learning algorithms to predict your chances of getting admitted to a university based on your academic and extracurricular profile.
    
    ### 🚀 Features
    - **Accurate Predictions**: Powered by state-of-the-art models like XGBoost and Random Forest.
    - **Comprehensive Analysis**: Evaluates GRE, TOEFL, CGPA, and qualitative factors like SOP and LOR.
    - **Personalized Recommendations**: Get actionable advice to improve your profile.
    - **Downloadable Reports**: Generate PDF reports of your prediction results.
    
    👈 **Select 'Admission Prediction' from the sidebar to get started!**
    """)

def prediction_page():
    st.markdown('<p class="sub-header">Student Admission Prediction</p>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.subheader("Academic Details")
        col1, col2, col3 = st.columns(3)
        with col1:
            gre = st.number_input("GRE Score (290-340)", min_value=290, max_value=340, value=310)
            sop = st.slider("SOP Strength (1.0-5.0)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        with col2:
            toefl = st.number_input("TOEFL Score (90-120)", min_value=90, max_value=120, value=100)
            lor = st.slider("LOR Strength (1.0-5.0)", min_value=1.0, max_value=5.0, value=3.0, step=0.5)
        with col3:
            cgpa = st.number_input("CGPA (6.8-10.0)", min_value=6.8, max_value=10.0, value=8.0, step=0.1)
            univ_rating = st.slider("Target Univ Rating (1-5)", min_value=1, max_value=5, value=3)
            
        st.subheader("Experience & Extracurriculars")
        col4, col5 = st.columns(2)
        with col4:
            research = st.selectbox("Research Experience", ["No", "Yes"])
            work_exp = st.number_input("Work Experience (Months)", min_value=0, max_value=120, value=0)
        with col5:
            extra_curr = st.selectbox("Extracurricular Activities", ["No", "Yes"])
            entrance = st.number_input("Entrance Exam Score (50-100)", min_value=50.0, max_value=100.0, value=75.0)
            
        st.subheader("Personal & Preferences")
        col6, col7 = st.columns(2)
        with col6:
            age = st.number_input("Age", min_value=18, max_value=40, value=22)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            category = st.selectbox("Category", ["General", "OBC", "SC", "ST"])
        with col7:
            courses = ['Computer Science', 'Data Science', 'Business Analytics', 'Electrical Engineering', 'Mechanical Engineering', 'Information Technology']
            universities = ['MIT', 'Stanford', 'Harvard', 'CMU', 'UC Berkeley', 'Oxford', 'Cambridge', 'ETH Zurich']
            pref_course = st.selectbox("Preferred Course", courses)
            pref_univ = st.selectbox("Preferred University", universities)
            
        submit_button = st.form_submit_button("Predict Admission Status")
        
    if submit_button:
        with st.spinner("Analyzing profile..."):
            input_data = {
                'GRE Score': gre,
                'TOEFL Score': toefl,
                'University Rating': univ_rating,
                'SOP': sop,
                'LOR': lor,
                'CGPA': cgpa,
                'Research Experience': 1 if research == "Yes" else 0,
                'Work Experience': work_exp,
                'Extracurricular Activities': 1 if extra_curr == "Yes" else 0,
                'Entrance Exam Score': entrance,
                'Category': category,
                'Gender': gender,
                'Age': age,
                'Preferred Course': pref_course,
                'Preferred University': pref_univ
            }
            
            prediction, confidence, recommendation = predict_admission(input_data)
            
            if prediction == "Error":
                st.error(recommendation)
            else:
                save_prediction(st.session_state.username, input_data, prediction, confidence)
                
                st.markdown("---")
                st.markdown('<p class="sub-header">Prediction Result</p>', unsafe_allow_html=True)
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    if prediction == "Admitted":
                        st.markdown(f'<p class="pred-admitted">🎉 {prediction}</p>', unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f'<p class="pred-rejected">❌ {prediction}</p>', unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center;'>Confidence: {confidence:.2f}%</h3>", unsafe_allow_html=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                with res_col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### AI Recommendation")
                    st.info(recommendation)
                    
                    # Generate PDF
                    pdf_path = generate_pdf_report(input_data, prediction, confidence, recommendation)
                    with open(pdf_path, "rb") as file:
                        btn = st.download_button(
                            label="📄 Download Full Report (PDF)",
                            data=file,
                            file_name="admission_report.pdf",
                            mime="application/pdf"
                        )
                    st.markdown('</div>', unsafe_allow_html=True)

def dataset_page():
    st.markdown('<p class="sub-header">Dataset Overview</p>', unsafe_allow_html=True)
    try:
        df = pd.read_csv('dataset/admission_dataset.csv')
        
        st.write("### Dataset Summary")
        st.write(f"**Total Records:** {df.shape[0]}")
        st.write(f"**Total Features:** {df.shape[1]}")
        
        st.dataframe(df.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Missing Values Report")
            st.dataframe(df.isnull().sum())
        with col2:
            st.write("### Basic Statistics")
            st.dataframe(df.describe())
            
        st.write("### Visualizations")
        
        fig1, fig2 = st.columns(2)
        with fig1:
            st.write("**Admission Distribution**")
            fig_dist = px.pie(df, names='Admission Status', color_discrete_sequence=['#10b981', '#ef4444'])
            st.plotly_chart(fig_dist, use_container_width=True)
            
        with fig2:
            st.write("**CGPA vs Admission**")
            fig_box = px.box(df, x='Admission Status', y='CGPA', color='Admission Status')
            st.plotly_chart(fig_box, use_container_width=True)
            
        st.write("**Correlation Heatmap**")
        numeric_df = df.select_dtypes(include=[np.number])
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        st.pyplot(fig_corr)
        
    except FileNotFoundError:
        st.warning("Dataset not found. Please ensure data is generated.")

def model_performance_page():
    st.markdown('<p class="sub-header">Model Performance</p>', unsafe_allow_html=True)
    try:
        results_df = pd.read_csv('models/model_performance.csv')
        st.dataframe(results_df.style.highlight_max(axis=0, subset=['Accuracy', 'F1-Score', 'ROC-AUC']))
        
        st.write("### Model Comparison (ROC-AUC)")
        fig = px.bar(results_df, x='Model', y='ROC-AUC', color='ROC-AUC', text_auto='.3f', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.warning("Model performance data not found. Please train the models first.")

def history_page():
    st.markdown('<p class="sub-header">My Prediction History</p>', unsafe_allow_html=True)
    df = get_user_history(st.session_state.username)
    if not df.empty:
        st.dataframe(df)
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History as CSV", data=csv, file_name="my_history.csv", mime="text/csv")
    else:
        st.info("No prediction history found.")

def admin_page():
    st.markdown('<p class="sub-header">Admin Dashboard</p>', unsafe_allow_html=True)
    df = get_all_history()
    if not df.empty:
        search = st.text_input("Search by Username")
        if search:
            df = df[df['username'].str.contains(search, case=False)]
            
        st.dataframe(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("### Predictions Overview")
            fig = px.pie(df, names='prediction', title='Total Predictions')
            st.plotly_chart(fig)
    else:
        st.info("No records found in database.")

def about_page():
    st.markdown('<p class="sub-header">About the Project</p>', unsafe_allow_html=True)
    st.write("""
    This project is an AI-Enabled Student Admission Prediction System.
    
    **Technologies Used:**
    - Frontend: Streamlit
    - Backend: Python, SQLite
    - Machine Learning: Scikit-Learn, XGBoost
    - Visualizations: Plotly, Seaborn, Matplotlib
    - Report Generation: fpdf2
    
    **Machine Learning Workflow:**
    Data Preprocessing -> Feature Scaling -> Model Training (LR, DT, RF, SVM, KNN, XGBoost) -> Evaluation -> Prediction.
    """)

def contact_page():
    st.markdown('<p class="sub-header">Contact Us</p>', unsafe_allow_html=True)
    st.write("For any queries, please reach out:")
    st.write("📧 Email: support@admissionai.com")
    st.write("🔗 GitHub: github.com/admissionai")

if __name__ == "__main__":
    main()
