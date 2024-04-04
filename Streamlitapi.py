import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
from io import BytesIO

# Function to load data from GitHub
def load_data(url):
    response = requests.get(url)
    data = response.content
    return pd.read_csv(BytesIO(data))

# Load the features (X)
df_url = 'https://raw.githubusercontent.com/chandrugtx8/housing-sales/main/Flat%20prices.csv'
df = load_data(df_url)
df['month'] = pd.to_datetime(df['month'])
df['year'] = df['month'].dt.year
df['month_of_year'] = df['month'].dt.month
df['remaining_lease_years'] = df['remaining_lease'].apply(lambda x: int(x.split()[0]))
df.drop(columns=['month', 'remaining_lease', 'block', 'street_name'], inplace=True)
X = pd.get_dummies(df, columns=['town', 'flat_type', 'storey_range', 'flat_model']).drop(columns=['resale_price'])

# Load the trained model
model_url = 'https://github.com/chandrugtx8/housing-sales/raw/main/random_forest_model2%20(1).pkl'
response = requests.get(model_url)
with open('random_forest_model.pkl', 'wb') as f:
    f.write(response.content)

model = pickle.load(BytesIO(response.content))

def main(X):
    st.title('Housing Sales Prediction')

    town = st.text_input("Enter the town: ")
    flat_type = st.text_input("Enter the flat type: ")
    storey_range = st.text_input("Enter the storey range: ")
    floor_area_sqm_str = st.text_input("Enter the floor area (in sqm): ")
    flat_model = st.text_input("Enter the flat model: ")
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

            input_data_encoded = pd.get_dummies(input_data).reindex(columns=X.columns, fill_value=0)
            predicted_price = model.predict(input_data_encoded)[0]
            st.success('Predicted Resale Price: {:.2f}'.format(predicted_price))
        except ValueError:
            st.error("Please enter valid numeric values for floor area and lease commence date.")

if __name__ == '__main__':
    main(X)

