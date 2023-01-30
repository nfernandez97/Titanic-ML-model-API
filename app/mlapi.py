from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
from starlette.responses import RedirectResponse

app = FastAPI()

class Passanger(BaseModel):
    Pclass: int #Int value (1,2,3)
    Sex: int # Int value 0=Female 1=Male
    Age: int # Int value
    SibSp: int # Int value
    Parch: int # Int value
    Fare: float # Float value
    Embarked: int # Int value 0=Queenstown 1=Southampton 2=Cherbourg


with open('model.pkl', 'rb') as f:
    model = pickle.load(f)    

@app.get('/')
def root():
    return RedirectResponse(url="/docs/")

@app.post('/predict')
def prediction(item:Passanger):
    df = pd.DataFrame([item.dict().values()],columns=item.dict().keys())
    pred = model.predict(df)
    if int(pred) == 1:
        pred = "Survived"
    else:
        pred = "Died"    
    return {"prediction":pred}