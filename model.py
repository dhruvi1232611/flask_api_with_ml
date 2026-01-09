import pandas as pd
import numpy as  np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import  joblib


df=pd.read_csv("income.csv")
x=df.drop(columns=['Income($)','Name'])
y=df['Income($)']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)

joblib.dump(model,'model.pkl')
print("Model is trained and successfully saved")