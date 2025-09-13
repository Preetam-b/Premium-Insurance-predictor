from pydantic import BaseModel, Field, ValidationError ,computed_field,field_validator 
from typing import List, Optional,Annotated,Literal
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
import pickle
import numpy as np  
import pandas as pd  
import os

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'mymodel.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
app = FastAPI()
tier_1_cities=['Mumbai','Delhi','Banglore','Chennai','Kolkata','Pune','Hyderabad','Ahemadabad']
tier_2_cities = [
    'Agra', 'Ajmer', 'Aligarh', 'Amravati', 'Amritsar', 'Anand', 'Asansol', 'Aurangabad', 'Bareilly',
    'Belgaum', 'Bhavnagar', 'Bhopal', 'Bhubaneswar', 'Bikaner', 'Bokaro', 'Chandigarh', 'Coimbatore',
    'Cuttack', 'Dehradun', 'Dhanbad', 'Erode', 'Faridabad', 'Gaya', 'Ghaziabad', 'Gorakhpur',
    'Guntur', 'Gurgaon', 'Guwahati', 'Gwalior', 'Hubli-Dharwad', 'Indore', 'Jabalpur', 'Jaipur',
    'Jalandhar', 'Jammu', 'Jamnagar', 'Jamshedpur', 'Jhansi', 'Jodhpur', 'Kanpur', 'Karnal',
    'Kochi', 'Kolhapur', 'Kollam', 'Kota', 'Kozhikode', 'Ludhiana', 'Lucknow', 'Madurai', 'Mangalore',
    'Meerut', 'Moradabad', 'Mysore', 'Nagpur', 'Nashik', 'Nellore', 'Noida', 'Palakkad', 'Patna',
    'Pondicherry', 'Prayagraj', 'Raipur', 'Rajahmundry', 'Rajkot', 'Ranchi', 'Rourkela', 'Salem',
    'Sangli', 'Siliguri', 'Solapur', 'Srinagar', 'Surat', 'Thiruvananthapuram', 'Thrissur',
    'Tiruchirappalli', 'Tirunelveli', 'Ujjain', 'Vadodara', 'Varanasi', 'Vasai-Virar', 'Vijayawada',
    'Visakhapatnam', 'Warangal'
]
class  UserInput(BaseModel):
    age:Annotated[int,Field(...,gt=0,lt=120,description='Age Of the User')]
    weight:Annotated[float,Field(...,gt=0,description='Weight of the User in kg')]
    height:Annotated[float,Field(...,gt=0,lt=2.5,description='Height of the User in meters ')]
    income:Annotated[float,Field(...,gt=0,description='Income of the User in lpa')]
    smoker:Annotated[bool,Field(...,description="Is the user smoker?")]
    city:Annotated[str,Field(...,description="City where he lives ")]
    occupation: Annotated[Literal['retired', 'freelancer', 'student', 'government_job',
       'business_owner', 'unemployed', 'private_job'], Field(..., description='Occupation of the user')]
    
    @field_validator('city')
    @classmethod
    def normalize_city(cls,v:str)->str:
       v=v.strip().title()
       return v
    @computed_field
    @property
    def bmi(self)->float:
       return self.weight/(self.height**2)
    @computed_field
    @property
    def lifestyle_risk(self)->str:
       if self.smoker or self.bmi>30:
         return "high"
       elif self.bmi<18.5:
         return "medium"
       else:
         return "low"
    @computed_field
    @property
    def age_group(self)->str:
       if self.age<25:
         return "young"
       elif self.age<49:
         return "adult"
       elif self.age<60:
         return "middle_aged"
       else:
         return "senior"
    @computed_field
    @property
    def city_tier(self)->int:
       if self.city in tier_1_cities:
          return 1
       elif self.city in tier_2_cities:
          return 2
       else:
          return 3
@app.post('/predict')
def predict_premium(data:UserInput):
    # Map new occupation categories to old ones
    occupation_mapping = {
        'government_job': 'government_employee',
        'private_job': 'private_employee',
        'business_owner': 'self_employed',
        'freelancer': 'self_employed',
        'unemployed': 'student',  # mapping unemployed to closest category
        'student': 'student',
        'retired': 'retired'
    }

    # Map the occupation to the model's expected categories
    mapped_occupation = occupation_mapping.get(data.occupation, 'private_employee')  # default to private_employee if unknown

    input_df = pd.DataFrame({
        'bmi': [data.bmi],
        'age_group': [data.age_group],
        'lifestyle_risk': [data.lifestyle_risk],
        'city_tier': [data.city_tier],
        'income_lpa': [data.income],
        'occupation': [mapped_occupation]
    })
    
    # Get prediction
    prediction = model.predict(input_df)[0]
    
    # Map the prediction from basic/premium to low/medium/high
    if prediction.lower() == 'basic':
        if data.income > 10 or data.city_tier == 1:
            category = 'medium'
        else:
            category = 'low'
    else:  # premium
        if data.lifestyle_risk == 'high' or data.income > 20:
            category = 'high'
        else:
            category = 'medium'
            
    return JSONResponse(
        status_code=200, 
        content={
            'predicted_category': category,
            'description': f'Insurance premium category is predicted to be {category.upper()}'
        }
    )

@app.get('/')
def home():
    return {'message':'Insurance Prediction API'}

@app.get('/health')
def health_check():
    return {'status':'OK',
    'version':'1.0.0',
    'model_loaded':model is not None}