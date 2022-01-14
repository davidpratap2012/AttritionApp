# Step1: import the modules and the model
import pycaret
import streamlit as st
import pandas as pd
from pycaret.classification import load_model, predict_model

model=load_model('Final_model')

# Step 2 create a prediction function to get the prediction given the users input
def predict(model, input_df):
    pred_df=predict_model(estimator=model, data=input_df)
    prediction=pred_df['Label'][0]
    return prediction

# The App starts here
st.title('Employee Attrition Prediction APP')

# Step3: create the fields to get the users input from front end 
satisfaction_level=st.number_input('satisfaction_level', min_value=0.1, max_value=1.0, value=0.1)
last_evaluation=st.number_input('last_evaluation', min_value=0.1, max_value=1.0, value=0.1)
number_project=st.slider('number_project', min_value=0, max_value=50)
average_montly_hours=st.slider('average_montly_hours', min_value=100, max_value=400, value=100)
time_spend_company=st.number_input('time_spend_company', min_value=1, max_value=20, value=1)
Work_accident=st.number_input('Work_accident', min_value=0, max_value=20, value=0)
promotion_last_5years=st.number_input('promotion_last_5years', min_value=0, max_value=20, value=0)
department=st.selectbox('department', ['sales', 'accounting', 'hr', 'technical', 'support', 'management',\
       'IT', 'product_mng', 'marketing', 'RandD'])
salary=st.selectbox('salary', ['low', 'medium', 'high'])

# Step4: Declare the input dataframe and output variable

input_dict={
    'satisfaction_level':satisfaction_level, 
    'last_evaluation':last_evaluation, 
    'number_project':number_project,
    'average_montly_hours':average_montly_hours, 
    'time_spend_company':time_spend_company, 
    'Work_accident':Work_accident,
    'promotion_last_5years':promotion_last_5years, 
    'department':department, 
    'salary':salary
}

input_df=pd.DataFrame([input_dict])
output=""

# Step5: Get the prediction and interpretation 
if st.button('Predict'):
    output=predict(model=model, input_df=input_df)
    output=str(output)
    
if output=='1':
    st.success('The employee will leave the company')
else: 
    st.success('The employee stays')
    
# END of the APP
