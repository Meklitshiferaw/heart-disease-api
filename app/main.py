from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.model import logistic_model, decision_tree_model

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HeartInput(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

def format_input(data: HeartInput):
    return [[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]]

@app.post("/predict/logistic")
def predict_logistic(data: HeartInput):
    values = format_input(data)
    prediction = logistic_model.predict(values)[0]
    return {"model": "Logistic Regression", "prediction": int(prediction)}


@app.post("/predict/tree")
def predict_tree(data: HeartInput):
    values = format_input(data)
    prediction = decision_tree_model.predict(values)[0]
    return {"model": "Decision Tree", "prediction": int(prediction)}
