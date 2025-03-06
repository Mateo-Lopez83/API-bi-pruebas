from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException
from DataModel import DataModel
from pydantic import BaseModel
from joblib import load
import numpy as np
from custom import transform_mjd, convert_ra_dec, clean_class_column
from joblib import load


app = FastAPI()


class RequestModel(BaseModel):
    data: List

@app.get("/")
def read_root():
   return {"Hello": "Test"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
async def make_predictions(payload: RequestModel):
    try:
        data = payload.data
        # Convert list to DataModel
        dataModel = DataModel(
            objid=data[0],
            ra= data[1],
            dec= data[2],
            u= data[3],
            g= data[4],
            r= data[5],
            i= data[6],
            z= data[7],
            run= data[8],
            camcol= data[9],
            field= data[10],
            score= data[11],
            clean= data[12],
            classStr= data[13],
            mjd= data[14],
            rowv= data[15],
            colv= data[16],
        )
        # Create DataFrame
        df = pd.DataFrame([dataModel.model_dump()])
        df.columns = dataModel.columns()
        print(df)
        # Load Model
        model = load("regression_pipeline.joblib")
        result = model.predict(df)

        return {"prediction": result.tolist()}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))