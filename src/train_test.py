import pandas as pd
from model import LinearRegression
from utils import preprocess_data, encode_data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=pd.read_csv("C:/Users/the43/Desktop/my projects/Learning AI/salary predection/data/jobs.csv")
processed = preprocess_data(data)
encoded_data = encode_data(processed)
X = encoded_data.drop(['Salary'], axis=1).values
y = encoded_data['Salary'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions)
print(y_test)
print("Accuracy:", model.score(X_test, y_test, predictions))