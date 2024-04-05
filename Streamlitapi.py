
import pickle
import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Load the features (X) from CSV file
@st.cache  # Cache the data to avoid reloading on every run
def load_data(csv_url):
    response = requests.get(csv_url)
    return pd.read_csv(BytesIO(response.content))

# Load the trained model from pickle file
@st.cache  # Cache the model to avoid reloading on every run
def load_model(pickle_url):
    response = requests.get(pickle_url)
    return pickle.load(BytesIO(response.content))

# Define URLs for CSV and pickle files
csv_url = 'https://raw.githubusercontent.com/chandrugtx8/housing-sales/main/Flat%20prices.csv'
pickle_url = 'https://raw.githubusercontent.com/chandrugtx8/housing-sales/main/random_forest_model2%20(1).pkl'

# Load data and model
df = load_data(csv_url)
model = load_model(pickle_url)

def main():
    st.title('Housing Sales Prediction')

    town_options = df['town'].unique()
    town = st.selectbox("Select the town:", town_options)

    flat_type_options = df['flat_type'].unique()
    flat_type = st.selectbox("Select the flat type:", flat_type_options)

    storey_range_options = df['storey_range'].unique()
    storey_range = st.selectbox("Select the storey range:", storey_range_options)

    floor_area_sqm_str = st.text_input("Enter the floor area (in sqm): ")
    
    flat_model_options = df['flat_model'].unique()
    flat_model = st.selectbox("Select the flat model:", flat_model_options)

    lease_commence_date_str = st.text_input("Enter the lease commence date: ")

    if st.button('Predict'):
        try:
            floor_area_sqm = float(floor_area_sqm_str)
            lease_commence_date = int(lease_commence_date_str)
            input_data = pd.DataFrame({
                'town': [town],
                'flat_type': [flat_type],
                'storey_range': [storey_range],
                'floor_area_sqm': [floor_area_sqm],
                'flat_model': [flat_model],
                'lease_commence_date': [lease_commence_date]
            })

            # Ensure input data is encoded in the same way as training data
            input_data_encoded = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)

            # Make prediction
            predicted_price = model.predict(input_data_encoded)[0]
            st.success('Predicted Resale Price: {:.2f}'.format(predicted_price))
        except ValueError:
            st.error("Please enter valid numeric values for floor area and lease commence date.")

if __name__ == '__main__':
    main()

