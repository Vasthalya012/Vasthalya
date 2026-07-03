import sqlite3
import bcrypt
import pandas as pd
from fpdf import FPDF
import os

DB_NAME = 'database.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    # Create Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            is_admin BOOLEAN
        )
    ''')
    # Create Predictions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            gre_score REAL,
            toefl_score REAL,
            cgpa REAL,
            course TEXT,
            university TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Add a default admin if none exists
    c.execute("SELECT * FROM users WHERE username='admin'")
    if not c.fetchone():
        hashed = bcrypt.hashpw('admin123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", ('admin', hashed, True))
        
    conn.commit()
    conn.close()

def register_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)", (username, hashed, False))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT password, is_admin FROM users WHERE username=?", (username,))
    row = c.fetchone()
    conn.close()
    if row:
        hashed_password = row[0]
        is_admin = row[1]
        if bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
            return True, is_admin
    return False, False

def save_prediction(username, input_dict, prediction, confidence):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (username, gre_score, toefl_score, cgpa, course, university, prediction, confidence)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        username,
        input_dict.get('GRE Score'),
        input_dict.get('TOEFL Score'),
        input_dict.get('CGPA'),
        input_dict.get('Preferred Course'),
        input_dict.get('Preferred University'),
        prediction,
        confidence
    ))
    conn.commit()
    conn.close()

def get_user_history(username):
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predictions WHERE username=?", conn, params=(username,))
    conn.close()
    return df

def get_all_history():
    conn = sqlite3.connect(DB_NAME)
    df = pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", conn)
    conn.close()
    return df

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'AI-Enabled Student Admission Prediction Report', 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(input_dict, prediction, confidence, recommendation, filename="prediction_report.pdf"):
    pdf = PDF()
    pdf.add_page()
    
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Student Profile:', 0, 1)
    pdf.set_font('Arial', '', 11)
    for key, value in input_dict.items():
        pdf.cell(0, 8, f"{key}: {value}", 0, 1)
        
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'Prediction Result:', 0, 1)
    pdf.set_font('Arial', '', 11)
    
    # Check prediction status and assign color
    if prediction == 'Admitted':
        pdf.set_text_color(0, 150, 0)
    else:
        pdf.set_text_color(200, 0, 0)
        
    pdf.cell(0, 8, f"Status: {prediction}", 0, 1)
    pdf.set_text_color(0, 0, 0)
    
    pdf.cell(0, 8, f"Confidence: {confidence:.2f}%", 0, 1)
    
    pdf.ln(5)
    pdf.set_font('Arial', 'B', 12)
    pdf.cell(0, 10, 'AI Recommendation:', 0, 1)
    pdf.set_font('Arial', '', 11)
    pdf.multi_cell(0, 8, recommendation)
    
    os.makedirs('static', exist_ok=True)
    filepath = os.path.join('static', filename)
    pdf.output(filepath)
    return filepath
