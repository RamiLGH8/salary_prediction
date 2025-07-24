import pandas as pd
import numpy as np

def preprocess_data(data):
    
    data.drop(columns=['Age', 'Gender'], axis=1, inplace=True)

    data['Salary'].fillna(data['Salary'].mean(), inplace=True)
    
    data['Years of Experience'].fillna(data['Years of Experience'].mode()[0], inplace=True)

    data['Education Level'].fillna(data['Education Level'].mode()[0], inplace=True)

    data['Job Title'].fillna(data['Job Title'].mode()[0], inplace=True)
   
    
    return data

def encode_data(data):
     min_salary = data['Salary'].min()
     max_salary = data['Salary'].max()
     min_experience = data['Years of Experience'].min()
     max_experience = data['Years of Experience'].max()
     data['education_level_encoded']=data['Education Level'].map({"Bachelor's": 0, "Master's": 1, 'PhD': 2})
     data["job_title_encoded"] = (data["Job Title"].map(data.groupby("Job Title")["Salary"].mean())- min_salary) / (max_salary - min_salary)
     data['experence_level_encoded'] = (data['Years of Experience']- min_experience) / (max_experience - min_experience)
     data.drop(columns=['Education Level', 'Job Title', 'Years of Experience'], inplace=True)
   
     return data



def train_test_split(X, y, test_size=0.2):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split = int((1 - test_size) * len(X))
    return X[indices[:split]], X[indices[split:]], y[indices[:split]], y[indices[split:]]



