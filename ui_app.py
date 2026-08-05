# ===============================
# IMPORTS
# ===============================
import streamlit as st
import numpy as np
import pickle
import pywt
from scipy.stats import entropy
from scipy.fft import fft
import mysql.connector
import hashlib
import re
import os

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="EEG Seizure Detection",
    page_icon="🧠",
    layout="wide"
)

# ===============================
# CONSTANTS
# ===============================
REQUIRED_MANUAL_SAMPLES = 45
REQUIRED_FILE_SAMPLES = 4097
MODEL_PATH = r"C:\main project\saved_model.pkl"

# ===============================
# SESSION STATE
# ===============================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "manual_signal" not in st.session_state:
    st.session_state.manual_signal = None

if "file_signal" not in st.session_state:
    st.session_state.file_signal = None

# ===============================
# HEADER
# ===============================
st.title("🧠 EEG Seizure Detection System")

# ===============================
# THEME TOGGLE
# ===============================
col1, col2 = st.columns([8, 2])
with col2:
    theme = st.selectbox("Theme", ["Light Mode", "Dark Mode"])

if theme == "Dark Mode":
    st.markdown("""
        <style>
        .stApp { background-color: #0e1117; color: white; }
        h1,h2,h3,h4,h5,h6,p,label { color: white !important; }
        .stTextInput>div>div>input,
        .stTextArea textarea {
            background-color: #1c1f26;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

# ===============================
# GREEN BUTTON STYLE
# ===============================
st.markdown("""
<style>
div.stButton > button {
    background-color: #00C851;
    color: white;
    font-weight: bold;
    border-radius: 8px;
}
div.stButton > button:hover {
    background-color: #00a844;
}
</style>
""", unsafe_allow_html=True)

# ===============================
# DATABASE CONNECTION
# ===============================
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="system",
        database="seizure_db"
    )

# ===============================
# PASSWORD FUNCTIONS
# ===============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def is_valid_password(password):
    pattern = r'^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[@$!%*?&]).{8,}$'
    return re.match(pattern, password)

# ===============================
# REGISTER
# ===============================
def register_user():
    st.subheader("📝 Register")
    username = st.text_input("Username", key="reg_user")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_pass")

    if st.button("Register"):
        if not username or not email or not password:
            st.warning("⚠ All fields required")
            return

        if not is_valid_password(password):
            st.error("❌ Password must contain 8+ chars, uppercase, lowercase, number & special char")
            return

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, hash_password(password))
            )
            conn.commit()
            st.success("✅ Registration successful")
        except mysql.connector.Error:
            st.error("⚠ Email already exists")
        finally:
            conn.close()

# ===============================
# LOGIN
# ===============================
def login_user():
    st.subheader("🔐 Login")
    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login"):
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM users WHERE email=%s AND password=%s",
            (email, hash_password(password))
        )
        user = cursor.fetchone()
        conn.close()

        if user:
            st.session_state.logged_in = True
            st.success("✅ Login successful")
            st.rerun()
        else:
            st.error("❌ Invalid credentials")

# ===============================
# AUTH PAGE
# ===============================
if not st.session_state.logged_in:
    choice = st.radio("Choose option", ["Login", "Register"])
    if choice == "Login":
        login_user()
    else:
        register_user()
    st.stop()

# ===============================
# LOGOUT
# ===============================
if st.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ===============================
# LOAD MODEL
# ===============================
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file not found.")
    st.stop()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ===============================
# FEATURE FUNCTIONS
# ===============================
def hjorth_parameters(signal):
    d1 = np.diff(signal)
    d2 = np.diff(d1)
    var0 = np.var(signal)
    var1 = np.var(d1)
    var2 = np.var(d2)

    mobility = np.sqrt(var1 / (var0 + 1e-10))
    complexity = np.sqrt(var2 / (var1 + 1e-10)) / (mobility + 1e-10)

    return var0, mobility, complexity

def wavelet_energy(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    return [np.sum(c**2) for c in coeffs]

def fft_power(signal):
    fft_vals = np.abs(fft(signal))[:len(signal)//2]
    return [
        np.sum(fft_vals[0:4]),
        np.sum(fft_vals[4:8]),
        np.sum(fft_vals[8:12]),
        np.sum(fft_vals[12:30]),
        np.sum(fft_vals[30:50])
    ]

def extract_features(signal):
    return [
        np.mean(signal),
        np.std(signal),
        np.max(signal),
        np.min(signal),
        np.sum(signal**2),
        *hjorth_parameters(signal),
        entropy(np.histogram(signal, bins=50, density=True)[0] + 1e-12),
        *wavelet_energy(signal),
        *fft_power(signal)
    ]

def predict_signal(signal):
    features = [extract_features(signal)]
    return model.predict(features)[0]

# ===============================
# EEG INPUT
# ===============================
st.subheader("📥 EEG Input")

uploaded_file = st.file_uploader("Upload EEG File", ["txt", "csv"])

if uploaded_file:
    try:
        signal = np.loadtxt(uploaded_file, delimiter=",")
        if len(signal) == REQUIRED_FILE_SAMPLES:
            st.session_state.file_signal = signal
            st.success("✅ Valid EEG file")
            st.line_chart(signal[:500])
        else:
            st.error("❌ File must contain 4097 samples")
    except:
        st.error("❌ Invalid file format")

manual_input = st.text_area("Manual EEG Input (comma-separated, minimum 45 values)")

if manual_input.strip():
    try:
        values = [float(x.strip()) for x in manual_input.split(",") if x.strip()]
        if len(values) >= REQUIRED_MANUAL_SAMPLES:
            st.session_state.manual_signal = np.array(values)
            st.success("✅ Manual input accepted")
            st.line_chart(st.session_state.manual_signal)
        else:
            st.error("❌ Minimum 45 values required")
    except:
        st.error("❌ Invalid manual input")

# ===============================
# PREDICTION (FIXED)
# ===============================
if st.button("Predict Seizure"):

    signal = None

    if st.session_state.file_signal is not None:
        signal = st.session_state.file_signal
    elif st.session_state.manual_signal is not None:
        signal = st.session_state.manual_signal

    if signal is None:
        st.warning("⚠ Provide valid EEG data first")
    else:
        result = predict_signal(signal)

        if result == 1:
            st.error("⚠ Seizure Detected")
            st.info("💡 Contact a neurologist immediately.")
        else:
            st.success("✅ No Seizure Detected")
            st.info("💡 Continue regular monitoring.")