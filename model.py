# -*- coding: utf-8 -*-
"""
Created on Thu Oct  6 19:05:53 2022

@author: Pierre
"""

import pandas as pd


def preprocess_data(df):
    df = df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis = 1)
    
    df = df.fillna(df.mean())
    df = pd.get_dummies(df)
    
    return df

    