import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
df=pd.read_csv("data/titanic_toy.csv")
print(df.head())

print(df.isnull().sum())
df['Age'].fillna(df['Age'].mean(),inplace=True)
df['Fare'].fillna(df['Fare'].median(),inplace=True)

x=df.drop(columns=['Survived'])
y=df['Survived']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

model=LogisticRegression()
model.fit(x_train,y_train)
pred=model.predict(x_test)
accuracy=accuracy_score(y_test,pred)*100

print("Accuracy:",accuracy)

joblib.dump(model, "save_model/model1.pkl")
print("Model Saved Successfully")