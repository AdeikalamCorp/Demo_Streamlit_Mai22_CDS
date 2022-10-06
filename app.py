# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 18:56:48 2022

@author: Pierre
"""

import streamlit as st
import pandas as pd
from model import preprocess_data

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

page = st.sidebar.radio(label = "deiajd", options = [1, 2, 3])

df = pd.read_csv("https://assets-datascientest.s3.eu-west-1.amazonaws.com/train.csv")
df = preprocess_data(df)

if page == 1:
    st.title("Démonstration Streamlit - Mai22 CDS")
    
    st.markdown("""
                Cette application a été développée pour la promotion Mai22 CDS.
                
                Nous allons évaluer des modèles de ML entraînés sur le jeu de données
                du Titanic.
                
                """)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fd/RMS_Titanic_3.jpg/1280px-RMS_Titanic_3.jpg")
    
    st.write("Voici un aperçu du jeu de données:")
    
    st.write(df)
    
if page == 2: 

    st.markdown("Dans cette page, vous pourrez évaluer différentes modèles de ML sur le dataset.")
    X = df.drop("Survived", axis = 1)
    y = df['Survived']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    
    
    selection = st.radio(label = "Choisissez un modèle à évaluer:",
                         options = ["Logistic Regression",
                                    "Decision Tree",
                                    "KNN"])
    
    if selection == "Logistic Regression":
        model = LogisticRegression().fit(X_train, y_train)
        st.write(model.score(X_test, y_test))
        
    if selection == "Decision Tree":
        model = DecisionTreeClassifier().fit(X_train, y_train)
        st.write(model.score(X_test, y_test))
        
    if selection == "KNN":
        model = KNeighborsClassifier().fit(X_train, y_train)
        st.write(model.score(X_test, y_test))
    
    












