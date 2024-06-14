#streamlit run streamlit.py
import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Trained model
with open('best_model.pkl', 'rb') as model_file:
    best_model = pickle.load(model_file)

# Load label encoders
label_encoders = {}
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
for col in categorical_cols:
    le = LabelEncoder()
    le.classes_ = np.load(f'label_encoder_{col}.npy', allow_pickle=True)
    label_encoders[col] = le

# Selected features
selected_features = np.load('selected_features.npy', allow_pickle=True).tolist()

# Streamlit app
st.title("Bank Term Deposit Prediction")

st.sidebar.header("Client Information")
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=30, step=1)

# Display category names in the dropdowns
job = st.sidebar.selectbox("Job", options=list(label_encoders['job'].classes_))
marital = st.sidebar.selectbox("Marital Status", options=list(label_encoders['marital'].classes_))
education = st.sidebar.selectbox("Education", options=list(label_encoders['education'].classes_))
default = st.sidebar.radio("Default", options=list(label_encoders['default'].classes_))
housing = st.sidebar.radio("Housing Loan", options=list(label_encoders['housing'].classes_))
loan = st.sidebar.radio("Personal Loan", options=list(label_encoders['loan'].classes_))
contact = st.sidebar.selectbox("Contact", options=list(label_encoders['contact'].classes_))
month = st.sidebar.selectbox("Last Contact Month", options=list(label_encoders['month'].classes_))
day_of_week = st.sidebar.selectbox("Last Contact Day of Week", options=list(label_encoders['day_of_week'].classes_))

# Hidden features
duration = st.sidebar.slider("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=100, step=10)
campaign = st.sidebar.slider("Number of Contacts during this Campaign", min_value=1, max_value=50, value=1, step=1)
pdays = st.sidebar.slider("Days since last Contact", min_value=-1, max_value=1000, value=0, step=1)
previous = st.sidebar.slider("Number of Contacts before this Campaign", min_value=0, max_value=50, value=0, step=1)
poutcome = st.sidebar.selectbox("Outcome of Previous Campaign", options=list(label_encoders['poutcome'].classes_))
emp_var_rate = st.sidebar.slider("Employment Variation Rate", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
cons_price_idx = st.sidebar.slider("Consumer Price Index", min_value=90.0, max_value=100.0, value=93.0, step=0.1)
cons_conf_idx = st.sidebar.slider("Consumer Confidence Index", min_value=-50.0, max_value=0.0, value=-40.0, step=0.1)
euribor3m = st.sidebar.slider("Euribor 3 Month Rate", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
nr_employed = st.sidebar.slider("Number of Employees", min_value=0, max_value=100000, value=5000, step=100)

# User input
user_input = {
    'age': age,
    'job': job,
    'marital': marital,
    'education': education,
    'default': default,
    'housing': housing,
    'loan': loan,
    'contact': contact,
    'month': month,
    'day_of_week': day_of_week,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed
}

# Convert to DataFrame
user_data = pd.DataFrame(user_input, index=[0])

# Show raw input data
if st.checkbox("Show raw input data"):
    st.subheader("Raw Input Data")
    st.write(user_input)

# Encode categorical variables
encoded_user_data = user_data.copy()
for col in categorical_cols:
    if col in encoded_user_data.columns:
        encoded_user_data[col] = label_encoders[col].transform(encoded_user_data[col])

# Ensure column order
encoded_user_data = encoded_user_data[selected_features]

# Predict button
if st.button("Predict"):
    prediction = best_model.predict(encoded_user_data)

    # Display prediction
    st.subheader("Prediction Result")
    st.write("Will the client subscribe to a term deposit? :", "Yes" if prediction == 1 else "No")
