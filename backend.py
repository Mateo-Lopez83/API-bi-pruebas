import pandas as pd
from fastapi import FastAPI, HTTPException
from DataModel import DataModel
from joblib import load
from joblib import load


app = FastAPI()
model_pipeline = load("prueba_modelo.joblib")

@app.get("/")
def read_root():
   return {"Hello": "ha"}

@app.post("/predict")
def make_predictions(dataModel: DataModel):
    try:
        df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
        df.columns = dataModel.columns()

        model = load("prueba_modelo.joblib")
        result = model.predict(df)
        
        result_list = result.tolist() if hasattr(result, 'tolist') else result
        
        return {"prediction":result_list}
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))
    

@app.get("/analytics")
def get_analytics():
    try:
        metrics = model_pipeline.evaluate()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))