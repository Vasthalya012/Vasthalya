# AI-Enabled Student Admission Prediction System

A complete Machine Learning and Streamlit-based web application to predict student admission chances based on academic scores, extracurriculars, and personal details.

## Features
- **Machine Learning Models**: Logistic Regression, Decision Tree, Random Forest, SVM, KNN, XGBoost, Gradient Boosting.
- **Data Preprocessing**: Handles missing values, performs scaling (StandardScaler), and One-Hot Encoding.
- **Interactive Web App**: Built with Streamlit for a modern, responsive user experience.
- **User Authentication**: Secure login/registration using bcrypt and SQLite.
- **PDF Report Generation**: Downloadable prediction results and recommendations using `fpdf2`.
- **Admin Dashboard**: View all historical predictions and search/filter.
- **Interactive Visualizations**: Correlation heatmaps, feature distributions, and model performance comparisons using Plotly and Seaborn.

## Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/admission-system.git
   cd admission-system
   ```

2. **Create a Virtual Environment (Optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate Synthetic Dataset**
   ```bash
   python generate_dataset.py
   ```
   *(This will create a synthetic dataset in the `dataset/` directory)*

5. **Train the Models**
   ```bash
   python train_model.py
   ```
   *(This will train all ML models, select the best one based on ROC-AUC, and save the artifacts in the `models/` directory)*

6. **Run the Application**
   ```bash
   streamlit run app.py
   ```

## Default Admin Credentials
- **Username**: `admin`
- **Password**: `admin123`

## Deployment

### Deploy on Streamlit Cloud
1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Click "New app", select your repository, branch, and set the main file path to `app.py`.
4. Click "Deploy".

### Deploy on Render
1. Create a `render.yaml` or directly create a new Web Service on Render.
2. Connect your GitHub repository.
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `streamlit run app.py --server.port $PORT`

## Project Structure
- `app.py`: Main Streamlit application.
- `preprocessing.py`: Data loading, cleaning, scaling, and encoding.
- `train_model.py`: Model training, evaluation, and saving best model.
- `generate_dataset.py`: Synthetic dataset generator.
- `prediction.py`: Loads models and runs inference.
- `utils.py`: Database connection, user auth, history management, PDF generation.
- `requirements.txt`: Python dependencies.

## License
MIT License
