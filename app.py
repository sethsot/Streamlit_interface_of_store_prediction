# Load key libraries and packages
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import joblib


# Define key functions
# Function to load toolkit
def load_toolkit(relative_path):
     loaded_tk = joblib.load(relative_path)
     return loaded_tk
          
# Load the toolkit
loaded_toolkit = load_toolkit("src/exports.pkl")

# Function for the dataset
@st.cache_resource()
def load_data(relative_path):
     data=pd.read_csv(rel_path)
     return data

#Load the dataset
rel_path=r"data\clean_test.csv"
loaded_df=load_data(rel_path)

# Set app title
st.title('Predicting store sales')

# Define app sections
header = st.container()
dataset = st.container()
prediction = st.container()

# Set up the 'header' section
with header:
      header.write("This app is to predict store sales")
      header.write("---")

#Set up the 'dataset' section
with dataset:
      if dataset.checkbox('Preview the dataset'):
           dataset.write(loaded_df.head())
           dataset.markdown('Kindly check the sidebar for information on the dataset')
           dataset.write('---')

# Set up the 'sidebar'
st.sidebar.header('Navigation')
menu = ['About the App', 'About the Columns']
choice=st.sidebar.selectbox('Select an option', menu)

# Content for 'About the App'
if choice == 'About the App':
    st.sidebar.header('About the App')
    st.sidebar.write("""
    This is a Streamlit app designed to showcase information on sales forcast.
    """)

# Content for 'About the Columns'
elif choice == 'About the Columns':
    st.sidebar.header('About the Columns')
    st.sidebar.write("""
    This section provides information about the columns or data displayed in this app.
    """)

# Define the form
form = st.form(key = "Information", clear_on_submit=True)

# Create key lists
expected_inputs = ["oil_prices","store_nbr","onpromotion","store_cluster","Year","Month","Day","ProductCategory","city","state","type_y","IsHoliday","Holiday_transferred"]
categoricals = ["ProductCategory","city","state","type_y","IsHoliday","Holiday_transferred"]
numerics = ["oil_prices","store_nbr","onpromotion","store_cluster","Year","Month","Day"]

# Set up the 'prediction' section
with prediction:
     prediction.subheader('Inputs')
     prediction.write('Kindly input information')

     # Define columns for user input fields
     left_col, right_col = prediction.columns(2)

     # Set up the form
     with form:
          
           #Set up the left column
          left_col.write('Inputs Part 1:')

          oil_prices = left_col.number_input('Enter crude oil prices', min_value= 0 )
          store_nbr = left_col.number_input('Enter store number', min_value=1, step=1, max_value=54)
          ProductCategory = left_col.selectbox('Select product category', options=['AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 'SEAFOOD'])
          onpromotion = left_col.select_slider('How manay products are on promotion?', options=range(0,719))
          IsHoliday = left_col.radio('Is your selected day a holiday Yes (Holiday) or No (not holiday)?', options=['Holiday','not holiday'])
          Holiday_transferred = left_col.radio('If your selected day was a holiday, was it transferred to another day True or False?', options=[True,False])
          city = left_col.selectbox('Chose a city', options=['Ambato', 'Babahoyo', 'Cayambe', 'Cuenca', 'Daule', 'El Carmen', 'Esmeraldas', 'Guaranda', 'Guayaquil', 'Ibarra', 'Latacunga', 'Libertad', 'Loja', 'Machala', 'Manta', 'Playas', 'Puyo', 'Quevedo', 'Quito', 'Riobamba', 'Salinas', 'Santo Domingo'])
                    
          # Set up the right column
          right_col.write('Inputs Part 2:')
          state = right_col.selectbox('Select a state', options=['Azuay', 'Bolivar', 'Chimborazo', 'Cotopaxi', 'El Oro', 'Esmeraldas', 'Guayas', 'Imbabura', 'Loja', 'Los Rios', 'Manabi', 'Pastaza', 'Pichincha', 'Santa Elena', 'Santo Domingo de los Tsachilas', 'Tungurahua'])
          type_y = right_col.selectbox('Select store type', options=['A','B','C','D','E'])
          store_cluster = right_col.number_input('Enter the cluster of store', min_value=1, step=1, max_value=17)
          Year = right_col.number_input('Enter year',min_value=2013, step=1)
          Month = right_col.number_input('Enter day of month', min_value=1, step=1, max_value=12)
          Day = right_col.number_input('Enter day of month', min_value=1, step=1, max_value=31)
          
                   
          #Create the submit button           
          submitted = form.form_submit_button('Predict')

 # Upon submission
if submitted:
            with prediction:
                 
                          
              # Format inputs
              input_dict = {
               'oil_prices':[oil_prices],
               'store_nbr':[store_nbr],
               'ProductCategory':[ProductCategory],
               'onpromotion':[onpromotion],
               'IsHoliday':[IsHoliday],
               'Holiday_transferred':[Holiday_transferred],
               'city':[city],
               'state':[state],
               'type_y':[type_y],
               'store_cluster':[store_cluster],
               'Year':[Year],
               'Month':[Month],
               'Day':[Day]                     
           }
            # Convert input data to a DataFrame
              input_data = pd.DataFrame.from_dict(input_dict)
              
              #Instantiate the pipeline
              model=loaded_toolkit["pipeline"]
          
            #Function to process inputs and return prediction
              def predict(model, input_data):
              
          # Convert inputs into a dataframe
                input_data=pd.DataFrame(model, input_data)

          # Make the prediction
                model_output = model.predict(input_data)
                #input_data["Prediction"] = model_output
              
                return model_output

             
            
              results=model.predict(input_data)
              rounded_results = np.round(results, 2)
              st.write(results)
              
              st.success(body=f"Prediction {rounded_results}")
            
