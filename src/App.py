import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
st.set_page_config(page_title='Obesity Risk Prediction', page_icon='🧊', layout='wide', initial_sidebar_state='auto')

st.header('Prediction of Obesity Risk')


def set_slide():
    st.sidebar.header("Upload your CSV data")
    inpute = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if inpute != None:
        # convring data to csv file
        inpute = pd.read_csv(inpute)
        # Display the DataFrame
        return inpute
    else:
        return None
st.write()

def continue_slide():
    gender = st.sidebar.slider('gender', min_value=12, max_value=100, value=25)
    age = st.sidebar.slider('age', min_value=12, max_value=100, value=25)
    height = st.sidebar.slider('height', min_value=1.0, max_value=3.0, value=1.8, step=.1)
    weight = st.sidebar.slider('weight', min_value=20, max_value=300, value=80, step=1)
    family_history = st.sidebar.selectbox('family_history', ['yes', 'no'])
    frequency_high_caloric_food = st.sidebar.selectbox('frequency_high_caloric_food', ['yes', 'no'])
    frequency_vegetables = st.sidebar.slider('frequency_vegetables', min_value=0.0, max_value=4.0, value=1.0, step=.1)
    main_meals = st.sidebar.slider('main_meals', min_value=0.0, max_value=4.0, value=1.0, step=.1)
    eating_out_main_meals = st.sidebar.selectbox('eating_out_main_meals', ['no', 'Sometimes', 'Frequently'])
    smoking = st.sidebar.selectbox('smoking', ['yes', 'no'])
    water_daily = st.sidebar.slider('water_daily', min_value=0.0, max_value=4.0, value=1.0, step=.1)
    calories_monitoring = st.sidebar.selectbox('calories_monitoring', ['yes', 'no'])
    technology_use = st.sidebar.slider('technology_use', min_value=0.0, max_value=4.0, value=1.0, step=.1)
    alcohol = st.sidebar.selectbox('alcohol', ['no', 'Sometimes', 'Frequently'])
    transportation = st.sidebar.selectbox('transportation', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])
    
    data = {
        'gender':gender,
        'age':age,
        'height':height,
        'weight':weight,
        'family_history':family_history,
        'frequency_high_caloric_food':frequency_high_caloric_food,
        'frequency_vegetables':frequency_vegetables,
        'main_meals':main_meals,
        'eating_out_main_meals':eating_out_main_meals,
        'smoking':smoking,
        'water_daily':water_daily,
        'calories_monitoring':calories_monitoring,
        'physical_activity':technology_use,
        'technology_use':technology_use,
        'alcohol':alcohol,
        'transportation':transportation
    }
    
    df = pd.DataFrame(data, index=[0])
    
    return df
        
data = set_slide()
taken_data = continue_slide()
def get(val):
    if val == 'Always':
        return 'Frequently'
    return val

def rename_columns(data, names):
    data.rename(columns=names, inplace=True)
    return data


def fix_data(df):
    columns_names = {
        "Gender": 'gender',
        'Age':"age",
        'Height':"height",
        'Weight':"weight",
        'family_history_with_overweight':"family_history",
        'FAVC':"frequency_high_caloric_food",
        'FCVC':"frequency_vegetables",
        'NCP':"main_meals",
        'CAEC':"eating_out_main_meals",
        'SMOKE':"smoking",
        'CH2O':"water_daily",
        'SCC':"calories_monitoring",
        'FAF':"physical_activity",
        'TUE':"technology_use",
        'CALC':"alcohol",
        'MTRANS':"transportation",
    }
    df = rename_columns(df, columns_names)
    df['alcohol'] = df['alcohol'].apply(get)
    return df

fix_data(data)

if data is not None:
    st.write("Uploaded DataFrame:", data)
    
    model = pickle.load(open("voting_classifier.pkl", "rb"))
    prediction = model.predict(data)
    
    data['prediction 📈'] = prediction
    
    st.write('### Prediction of the model for CSV file 📈: ')
    st.write(data)
