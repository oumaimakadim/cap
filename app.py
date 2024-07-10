import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from passlib.hash import pbkdf2_sha256
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from datetime import datetime
from streamlit_chat import message
import google.generativeai as palm
import pickle

API_KEY = 'AIzaSyBuc4pLvEoBKCDPoaUD_BPk7LchMAfvhXg'
palm.configure(api_key=API_KEY)

conn = sqlite3.connect('data.db')
c = conn.cursor()

def ensure_columns_exist():
    c.execute('PRAGMA table_info(form_data)')
    columns = [column[1] for column in c.fetchall()]
    required_columns = ['device_type', 'created_by', 'date_panne', 'date_installation', 'duree_utilisation', 'duree_panne', 'today_date']
    for col in required_columns:
        if col not in columns:
            c.execute(f'ALTER TABLE form_data ADD COLUMN {col} TEXT')
    conn.commit()

# Create tables if they don't exist
c.execute('''CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL
)''')
conn.commit()

c.execute('''CREATE TABLE IF NOT EXISTS form_data (
    id INTEGER PRIMARY KEY, 
    name TEXT, 
    satisfaction TEXT, 
    problem TEXT, 
    device_type TEXT, 
    component TEXT, 
    date_panne TEXT,
    date_installation TEXT,
    duree_utilisation INTEGER,
    duree_panne INTEGER,
    created_by TEXT,
    today_date TEXT
)''')
conn.commit()

ensure_columns_exist()

# Authentication Functions
def create_account():
    st.markdown('<div class="container"><h2>Create Account</h2></div>', unsafe_allow_html=True)
    username = st.text_input("Username", key="create_username")
    password = st.text_input("Password", type="password", key="create_password")
    if st.button("Create Account"):
        if username and password:
            hashed_password = pbkdf2_sha256.hash(password)
            try:
                c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, hashed_password))
                conn.commit()
                st.success("Account created successfully!")
            except sqlite3.IntegrityError:
                st.error("Username already taken. Please choose another username.")
        else:
            st.error("Please fill out all fields.")

def login():
    st.markdown('<div class="container"><h2>Login</h2></div>', unsafe_allow_html=True)
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Login"):
        if username and password:
            c.execute('SELECT password FROM users WHERE username = ?', (username,))
            result = c.fetchone()
            if result and pbkdf2_sha256.verify(password, result[0]):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
            else:
                st.error("Invalid username or password.")
        else:
            st.error("Please fill out all fields.")

# Form for data entry
def form():
    st.markdown('<div class="container"><h2>Fill the Form</h2></div>', unsafe_allow_html=True)
    name = st.text_input("Name")
    satisfaction = st.text_input("Satisfaction")
    # Allow selection of specific problems
    problem = st.selectbox("Problem", ["Need Replacement", "No Replacement Needed", "Other"])

    device_type = st.selectbox("Device Type", ["disjoncteur", "relais"])

    if device_type == "disjoncteur":
        components = [f"disjoncteur_component_{i}" for i in range(1, 11)]
    else:
        components = [f"relais_component_{i}" for i in range(1, 11)]
    
    component = st.selectbox("Component", components)

    date_panne = st.date_input("Date de panne")
    date_installation = st.date_input("Date d'installation du composant")
    today_date = datetime.today().date()

    # Calculate duree_utilisation automatically
    if date_panne and date_installation:
        duree_utilisation = (date_panne - date_installation).days
        st.write(f"Durée d'utilisation du composant (jours) : {duree_utilisation}")
    else:
        duree_utilisation = 0

    duree_panne = st.number_input("Durée de panne (minutes)", min_value=0)

    if st.button("Save"):
        save_data(name, satisfaction, problem, device_type, component, date_panne, date_installation, duree_utilisation, duree_panne, today_date)
        st.success("Data Saved")

# Function to save data to the database
def save_data(name, satisfaction, problem, device_type, component, date_panne, date_installation, duree_utilisation, duree_panne, today_date):
    created_by = st.session_state.username
    duree_utilisation = int(duree_utilisation)
    duree_panne = int(duree_panne)
    
    c.execute('''INSERT INTO form_data
                 (name, satisfaction, problem, device_type, component, date_panne, date_installation, duree_utilisation, duree_panne, created_by, today_date) 
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', 
              (name, satisfaction, problem, device_type, component, date_panne, date_installation, duree_utilisation, duree_panne, created_by, today_date))
    conn.commit()

# Function to get data from the database
def get_data():
    c.execute('SELECT * FROM form_data')
    return c.fetchall()

# Function to get column names from the database
def get_column_names():
    c.execute('PRAGMA table_info(form_data)')
    return [col[1] for col in c.fetchall()]

# View data
def view_data():
    st.markdown('<div class="container"><h2>View Data</h2></div>', unsafe_allow_html=True)
    data = get_data()
    if data:
        columns = get_column_names()
        df = pd.DataFrame(data, columns=columns)
        st.dataframe(df)

        selected_user = st.selectbox("Filter by User:", ["All"] + list(df['created_by'].unique()))
        if selected_user != "All":
            df = df[df['created_by'] == selected_user]
            st.dataframe(df)
    else:
        st.write("No data available")

# Data analysis and prediction
def analysis():
    st.markdown('<div class="container"><h2>Analysis and Prediction</h2></div>', unsafe_allow_html=True)
    data = get_data()
    if data:
        columns = get_column_names()
        df = pd.DataFrame(data, columns=columns)
        
        # Plotting number of problems by user
        st.markdown('<div class="container"><h3>Number of Problems by User</h3></div>', unsafe_allow_html=True)
        fig = px.bar(df, x='created_by', y='problem', color='created_by', title="Number of Problems by User", labels={'problem':'Number of Problems'})
        st.plotly_chart(fig)
        
        # Plotting number of breakdowns by component and duration
        st.markdown('<div class="container"><h3>Number of Breakdowns by Component and Duration</h3></div>', unsafe_allow_html=True)
        fig = px.bar(df, x='component', y='duree_panne', color='component', title="Number of Breakdowns by Component and Duration", labels={'duree_panne':'Duration of Breakdowns (minutes)'})
        st.plotly_chart(fig)
        
        # Plotting number of breakdowns by date
        st.markdown('<div class="container"><h3>Number of Breakdowns by Date</h3></div>', unsafe_allow_html=True)
        fig = px.line(df, x='date_panne', y='id', title="Number of Breakdowns by Date", labels={'id':'Number of Breakdowns'})
        st.plotly_chart(fig)
        
        # Prepare data for prediction
        df['date_installation'] = pd.to_datetime(df['date_installation'])
        df['date_panne'] = pd.to_datetime(df['date_panne'])
        df['today_date'] = pd.to_datetime(df['today_date'])
        df['Installation_to_Panne'] = (df['date_panne'] - df['date_installation']).dt.days
        
        X = df[['duree_utilisation', 'duree_panne', 'Installation_to_Panne']]
        y = (df['problem'] == 'Need Replacement').astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Build the neural network model
        model = Sequential()
        model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=10, validation_data=(X_test, y_test))

        score = model.evaluate(X_test, y_test, verbose=0)
        accuracy = score[1]

        st.markdown(f'<div class="container"><h3>Prediction Accuracy: {accuracy:.2f}</h3></div>', unsafe_allow_html=True)
        
        # New section for prediction based on a given date
        st.markdown('<div class="container"><h3>Predict Component Replacement by Date</h3></div>', unsafe_allow_html=True)
        predict_date = st.date_input("Enter the date for prediction")
        predict_date = pd.to_datetime(predict_date)
        st.markdown(f"Debug: Predict date - {predict_date}")

        components_to_replace = df[(df['date_installation'] <= predict_date) & (df['date_panne'] >= predict_date)]
        components_to_replace = components_to_replace[components_to_replace['problem'] == 'Need Replacement']

        if not components_to_replace.empty:
            st.markdown(f"<div class='container'><h4>Component(s) to replace on {predict_date.date()}:</h4></div>", unsafe_allow_html=True)
            st.dataframe(components_to_replace[['device_type', 'component']])
        else:
            st.markdown(f"<div class='container'><h4>No components need replacement on the selected date.</h4></div>", unsafe_allow_html=True)
    else:
        st.write("No data available for analysis")

# Add sample data directly to the database for testing
def add_sample_data():
    sample_data = [
        ("Sample 1", "Satisfied", "Need Replacement", "disjoncteur", "disjoncteur_component_1", "2024-07-10", "2024-07-01", 9, 34, "test_user", "2024-07-12"),
        ("Sample 2", "Dissatisfied", "No Replacement Needed", "disjoncteur", "disjoncteur_component_2", "2024-07-10", "2024-07-02", 8, 45, "test_user", "2024-07-12"),
        ("Sample 3", "Neutral", "Need Replacement", "relais", "relais_component_1", "2024-07-10", "2024-07-03", 7, 30, "test_user", "2024-07-12"),
        ("Sample 4", "Satisfied", "No Replacement Needed", "relais", "relais_component_2", "2024-07-10", "2024-07-04", 6, 20, "test_user", "2024-07-12")
    ]

    for data in sample_data:
        c.execute('''INSERT INTO form_data
                     (name, satisfaction, problem, device_type, component, date_panne, date_installation, duree_utilisation, duree_panne, created_by, today_date) 
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''', data)
    conn.commit()

add_sample_data()

# Generative AI
def generative_ai():
    st.markdown('<div class="container"><h2>Generative AI Assistant</h2></div>', unsafe_allow_html=True)
    user_input = st.text_input("Ask something:")
    if st.button("Generate Response"):
        if user_input:
            response = palm.generate_text(prompt=user_input)
            # Display only the generated text (response.result)
            st.write(response.result) 
        else:
            st.error("Please enter a query.")

# Main function to control the flow
def main():
    
    st.set_page_config(page_title="Maintenance App", page_icon=":wrench:", layout="wide")
    
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    st.sidebar.image("cap.png")  # Add the top logo
    st.sidebar.title("Navigation")

    st.markdown("""
        <style>
            .css-1aumxhk {
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .css-1aumxhk img {
                width: 100%;
            }
            .sidebar .sidebar-content {
                padding-top: 0px;
            }
            .stButton > button {
                background-color: #1f77b4;
                color: white;
                border-radius: 10px;
                border: none;
                font-size: 16px;
                padding: 10px 24px;
                margin: 10px;
            }
            .stButton > button:hover {
                background-color: #1c6ea4;
                color: white;
            }
            .stRadio > label {
                display: none;
            }
            .stRadio > div {
                display: flex;
                flex-direction: column;
            }
            .stRadio > div > div {
                margin: 5px 0;
            }
            .stRadio > div > div > label {
                background-color: #1f77b4;
                color: white;
                border-radius: 10px;
                border: none;
                font-size: 16px;
                padding: 10px 24px;
                text-align: center;
                cursor: pointer;
            }
            .stRadio > div > div:hover > label {
                background-color: #1c6ea4;
            }
            .stRadio > div > div:has(input:checked) > label {
                background-color: #1c6ea4;
            }
            .stRadio > div > div > input {
                display: none;
            }
        </style>
    """, unsafe_allow_html=True)

    if st.session_state.logged_in:
        st.sidebar.write(f"Logged in as: {st.session_state.username}")
        page = st.sidebar.radio("Go to", ["Fill Form", "View Data", "Analysis", "Generative AI Assistant", "Logout"], key="nav")
        if page == "Fill Form":
            form()
        elif page == "View Data":
            view_data()
        elif page == "Analysis":
            analysis()
        elif page == "Generative AI Assistant":
            generative_ai()
        elif page == "Logout":
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.success("Logged out successfully!")
    else:
        auth_choice = st.sidebar.radio("Authentication", ["Login", "Create Account"], key="auth")
        if auth_choice == "Login":
            login()
        elif auth_choice == "Create Account":
            create_account()

    st.sidebar.image("nora.png")  # Add the bottom logo

if __name__ == '__main__':
    main()
