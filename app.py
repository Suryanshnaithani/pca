import streamlit as st
import pandas as pd
import numpy as np
def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


install_and_import('joblib')

model = joblib.load('model.pkl')
def main():
    data = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
    data.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']
    st.title('Wine Classification')
    st.write('This is a wine classification web app')
    st.write('Please enter the required fields')
    alcohol = st.slider('Alcohol', float(data['Alcohol'].min()), float(data['Alcohol'].max()))
    malic_acid = st.slider('Malic acid', float(data['Malic acid'].min()), float(data['Malic acid'].max()))
    ash = st.slider('Ash', float(data['Ash'].min()), float(data['Ash'].max()))
    alcalinity_of_ash = st.slider('Alcalinity of ash', float(data['Alcalinity of ash'].min()), float(data['Alcalinity of ash'].max()))
    magnesium = st.slider('Magnesium', float(data['Magnesium'].min()), float(data['Magnesium'].max()))
    total_phenols = st.slider('Total phenols', float(data['Total phenols'].min()), float(data['Total phenols'].max()))
    flavanoids = st.slider('Flavanoids', float(data['Flavanoids'].min()), float(data['Flavanoids'].max()))
    nonflavanoid_phenols = st.slider('Nonflavanoid phenols', float(data['Nonflavanoid phenols'].min()), float(data['Nonflavanoid phenols'].max()))
    proanthocyanins = st.slider('Proanthocyanins', float(data['Proanthocyanins'].min()), float(data['Proanthocyanins'].max()))
    color_intensity = st.slider('Color intensity', float(data['Color intensity'].min()), float(data['Color intensity'].max()))
    hue = st.slider('Hue', float(data['Hue'].min()), float(data['Hue'].max()))
    od280_od315_of_diluted_wines = st.slider('OD280/OD315 of diluted wines', float(data['OD280/OD315 of diluted wines'].min()), float(data['OD280/OD315 of diluted wines'].max()))
    proline = st.slider('Proline', float(data['Proline'].min()), float(data['Proline'].max()))

    input_data = {'Alcohol': alcohol, 'Malic acid': malic_acid, 'Ash': ash, 'Alcalinity of ash': alcalinity_of_ash, 'Magnesium': magnesium, 'Total phenols': total_phenols, 'Flavanoids': flavanoids, 'Nonflavanoid phenols': nonflavanoid_phenols, 'Proanthocyanins': proanthocyanins, 'Color intensity': color_intensity, 'Hue': hue, 'OD280/OD315 of diluted wines': od280_od315_of_diluted_wines, 'Proline': proline}
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.write('Prediction:', prediction)
    st.write('Prediction Probability:', prediction_proba)

if __name__ == '__main__':
    main()
