import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Load Data with Error Handling

file_path = "burger_sales_analysis.csv"  # Ensure the file is in the same directory
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    st.error("Error: Data file not found. Please check the file path.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("Error: Data file is empty.")
    st.stop()
except pd.errors.ParserError:
    st.error("Error: Data file is corrupted.")
    st.stop()


# 2. Verify Required Columns

required_columns = ['Coca Cola Ordered', 'Pieces Sold', 'Year', 
                    'Selling Price (per piece)', 'Cost of Making (per piece)', 'Burger Name']
for col in required_columns:
    if col not in df.columns:
        st.error(f"Error: Missing required column '{col}' in dataset.")
        st.stop()


# 3. Data Preprocessing

# Fill missing values for relevant columns
df[['Coca Cola Ordered', 'Pieces Sold']] = df[['Coca Cola Ordered', 'Pieces Sold']].fillna(0)

# Replace 0 in 'Pieces Sold' to avoid division by zero (if any exist)
df['Pieces Sold'] = df['Pieces Sold'].replace(0, 1)

# Ensure 'Year' is numeric and has no NaN values
if df['Year'].isnull().sum() > 0 or not np.issubdtype(df['Year'].dtype, np.number):
    st.error("Error: 'Year' column contains missing or non-numeric values.")
    st.stop()


# 4. Train Models for Prediction

# Sales Prediction Model
sales_model = LinearRegression()
X = df[['Year']]
y_sales = df[['Pieces Sold']]
sales_model.fit(X, y_sales)

# Profit Prediction Model
df['Profit'] = (df['Selling Price (per piece)'] - df['Cost of Making (per piece)']) * df['Pieces Sold']
profit_model = LinearRegression()
y_profit = df[['Profit']]
profit_model.fit(X, y_profit)


# 5. Streamlit UI

st.title("🍔 Product Sales Analyzer")

# User Input: Year
year_input = st.number_input("Enter Year to Predict Sales & Profit", 
                             min_value=int(df['Year'].min()), 
                             max_value=int(df['Year'].max()) + 5, step=1)
year_input = int(year_input)  # Ensure input is integer

# Validate year input (optional check)
if year_input < df['Year'].min() or year_input > df['Year'].max() + 5:
    st.error(f"Error: Year {year_input} is out of prediction range.")

if st.button("Predict Sales & Profit"):
    # Perform predictions
    predicted_sales = sales_model.predict([[year_input]])[0][0]
    predicted_profit = profit_model.predict([[year_input]])[0][0]
    
    st.success(f"📈 Predicted Sales for {year_input}: {int(predicted_sales)} burgers")
    st.success(f"💰 Predicted Profit for {year_input}: ₹{int(predicted_profit)}")


# 6. Display Full Profit Analysis

st.subheader("📊 Full Profit Analysis")
st.write(df[['Burger Name', 'Year', 'Profit']])



#**********____________********************************



#  step 1 cd "C:\Users\khush\.ipython\Project AIML"
#  step 2 .\.venv\Scripts\activate 
#  step 3 pip install streamlit
#  step 4 pip install scikit-learn
#  step 5 streamlit run sales_analyzer_app.py
