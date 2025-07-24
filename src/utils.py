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



# data=pd.read_csv("C:/Users/the43/Desktop/my projects/Learning AI/salary predection/data/jobs.csv")
# processed = preprocess_data(data)
# encoded_data= encode_data(processed)
# print(encoded_data)
# print("Processed Data:", len(processed['Job Title'].unique())) 174
# print(len(processed['Education Level'].unique())) 3
# encoded_dat = encode_data(processed)
# # print(len(encoded_dat['employee_residence'].unique())) #104
# # print(len(encoded_dat['company_location'].unique())) #97
# print("Encoded Data:", encoded_dat, "\n" ,encoded_dat.dtypes)