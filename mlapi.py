from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from starlette.responses import RedirectResponse

app = FastAPI()

class Passanger(BaseModel): #Passanger class is created with its respective atributes
    Pclass: int #Int value (1,2,3)
    Sex: int # Int value 0=Female 1=Male
    Age: int # Int value
    SibSp: int # Int value
    Parch: int # Int value
    Fare: float # Float value
    Embarked: int # Int value 0=Queenstown 1=Southampton 2=Cherbourg


with open('model.pkl', 'rb') as f: #Load the trained model on a .pkl format
    model = pickle.load(f)    

@app.get('/')
def root():
    return RedirectResponse(url="/docs/") #redirect the root path to the fastapi path

@app.post('/predict')
def prediction(item:Passanger): #the function receives a passanger item
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys()) #Create a dataframe with the information on the JSON request
    pred = model.predict(df) #Predict 
    if int(pred) == 1: #traslate the prediction into words
        pred = "Survived"
    else:
        pred = "Died"    
    return {"prediction":pred}