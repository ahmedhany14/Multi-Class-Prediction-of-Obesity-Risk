
<br />
<p align="center">

  <h3 align="center"> Multi-Class-Prediction-of-Obesity-Risk </h3>
</p>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Description](#description)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Dataset](#Dataset)
- [Packages and frameworks i used](#packages-and-frameworks-i-used)
- [Installing packages](#installing-packages)
- [Deplyment and run the application](#deplyment-the-application)

## Description

This project aims to predict the risk of obesity using machine learning techniques. Obesity is a significant health concern globally, and early detection of obesity risk can help in preventing associated health complications.

## Methodology

I employ various machine learning algorithms such as Logistic Regression, Random Forest, and Support Vector Machines (SVM) to build predictive models. The performance of each model is evaluated using metrics such as accuracy, precision, recall, and F1-score.

## Repository Structure

* Data set: Contains the dataset used in the project
* Note book: Jupyter notebooks with the data preprocessing, model building, and evaluation code
* src: Python scripts for data preprocessing, model training, and evaluation.
* Models: Contains the trained model, that ready to use

## Dataset
The dataset used in this project contains various features such as demographic information, lifestyle factors, and health indicators. These features are used to predict the obesity risk level of individuals.

The dataset used in this project from Kaggle, to download it use this [link](https://www.kaggle.com/competitions/playground-series-s4e2/data)

## Packages and frameworks i used


* [numpy](https://keras.io/) for Algebra
* [pandas](https://pandas.pydata.org/docs/) for datasets
* [sklearn](https://scikit-learn.org/stable/index.html) for machine learning models and data cleaning 
* [streamlit](https://docs.streamlit.io/) for deployment
* [seaborn](https://seaborn.pydata.org/) and [matplotlib](https://matplotlib.org/) for data analysis and visualization and EDA
* [keras](https://keras.io/) for Neural Networks
* [pickle](https://docs.python.org/3/library/pickle.html)  converting a Python object into a byte stream to store it in a file


## Installing packages

#### open Terminal in VS code, and write following commands
        pip install pandas
        pip install sklearn
        pip install streamlit
        pip install seaborn
        pip install matplotlib
        pip install tensorflow
        pip install keras

## Deplyment and run the application

#### open Terminal in VS code, and write following commands
        cd "App"
        streamlit run App.py