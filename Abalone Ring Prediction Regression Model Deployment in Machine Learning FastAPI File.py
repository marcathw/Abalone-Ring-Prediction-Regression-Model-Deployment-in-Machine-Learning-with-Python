import uvicorn
from fastapi import FastAPI
import numpy as np
import pickle
from pydantic import BaseModel

app = FastAPI()

pickle_in = open("Abalone Ring Prediction Regression Model Deployment in Machine Learning Pickle File.pkl", "rb")
modelling = pickle.load(pickle_in)

class abalone_input(BaseModel):
    sex: str
    length: float
    diameter: float
    height: float
    whole_weight: float
    whole_weight_1: float
    whole_weight_2: float
    shell_weight: float

class PredictionResponse(BaseModel):
    prediction: int

@app.post("/predict", response_model = PredictionResponse)
def predict(data: abalone_input):
    try:
        sex_encoded = [0, 0, 0]
        if data.sex.upper() == "F":
            sex_encoded[0] = 1
        elif data.sex.upper() == "I":
            sex_encoded[1] = 1
        elif data.sex.upper() == "M":
            sex_encoded[2] = 1
        else:
            raise ValueError()

        input_data = [
            data.length,
            data.diameter,
            data.height,
            data.whole_weight,
            data.whole_weight_1,
            data.whole_weight_2,
            data.shell_weight
        ] + sex_encoded

        prediction = modelling.predict([input_data])
        return PredictionResponse(prediction = int(prediction[0]))

    except Exception as e:
        raise HTTPException(status_code = 500, detail = f"Prediction error: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port = 7100)
