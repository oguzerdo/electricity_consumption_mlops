from datetime import datetime
from fastapi import FastAPI, Depends, Request
from schemas import cons
import models
from database import engine, get_db
from sqlalchemy.orm import Session
import os
import mlflow
import pandas as pd
from scripts.preprocess_data import create_features
import json

def get_model(RUN_ID = "12d6e85e9b8d44cbb27a10779636dabe"):
    
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'
    
    logged_model = f"s3://vbo-mlflow-bucket/10/{RUN_ID}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)

    return model

app = FastAPI()

# Creates all the tables defined in models module
models.Base.metadata.create_all(bind=engine)

# Insert hourly prediction requests
def insert_houry_mwh(date, day, hour, prediction, client_ip, db):
    new_mwh = models.Consumption(
        type_ = "hourly",
        date_=date,
        day=day,
        hour=hour,
        prediction=prediction,
        client_ip=client_ip
    )

    db.add(new_mwh)
    db.commit()
    db.refresh(new_mwh)
    return new_mwh

# Insert daily prediction requests
def insert_daily_mwh(request, prediction, client_ip, db):
    
    new_mwh = models.Consumption(
        
        date_ = request["date_"],
        period = request["period"],
        type_ = "daily",
        prediction=prediction,
        client_ip=client_ip
    )

    db.add(new_mwh)
    db.commit()
    db.refresh(new_mwh)
    return new_mwh

# Hourly Prediction Func
def hourly_prediction_mwh(model_obj, date, day, hour):
    
    date_range = pd.date_range(date, periods=day*24 + hour, freq="H" )

    data_dict = {"Date": date_range,
                "MWh": None}

    dataframe = pd.DataFrame(data_dict)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe = dataframe.set_index("Date",drop=True)
    
    pred_data = create_features(dataframe)
    pred_data.drop("MWh", inplace=True, axis=1)

    prediction = model_obj.predict(pred_data)
    
    dataframe["MWh_pred"] = prediction
    pred_data = dataframe.reset_index(drop=True)
    pred_data["type"] = "hourly"
    
    return pred_data[["date", "MWh_pred", "type"]]

# Daily Predicton Func
def daily_prediction_mwh(model_obj, request):
    
    date = request["date_"]
    period = request["period"]
    date_range = pd.date_range(date, periods=period*24, freq="H")

    data_dict = {"Date": date_range,
                "MWh": None}

    dataframe = pd.DataFrame(data_dict)
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe = dataframe.set_index("Date",drop=True)

    pred_data = create_features(dataframe)
    pred_data.drop("MWh", inplace=True, axis=1)

    prediction = model_obj.predict(pred_data)

    dataframe["MWh_pred"] = prediction
    pred_data = dataframe.reset_index(drop=True)
    pred_data_group = pred_data.groupby(pred_data["date"].dt.date).agg({'MWh_pred':"sum"}).reset_index()
    pred_data_group["type"] = "daily"
    return pred_data_group[["date", "MWh_pred", "type"]]


# Daily MWh Consumption Prediction endpoint
@app.post("/daily")
def predict_mwh(request: cons, fastapi_req: Request,  db: Session = Depends(get_db)):
    prediction = daily_prediction_mwh(get_model(), request.dict())
    prediction = json.loads(prediction.to_json(orient="table",index=False))["data"]
    db_insert_record = insert_daily_mwh(request=request.dict(), prediction=prediction,
                                          client_ip=fastapi_req.client.host,
                                          db=db)
    return {"prediction": prediction, "db_record": db_insert_record}


# Hourly MWh Consumption Prediction endpoint
@app.post("/hourly/{date}/{day}/{hour}")
def predict_houry_mwh(date: datetime, day: int, hour: int, fastapi_req: Request,  db: Session = Depends(get_db)):
    
    prediction = hourly_prediction_mwh(get_model(), date, day, hour)
    prediction = json.loads(prediction.to_json(orient="table",index=False))["data"]
    db_insert_record = insert_houry_mwh(date=date, day=day, hour=hour, 
                                  prediction=prediction, client_ip=fastapi_req.client.host, db=db)
    return {"prediction": prediction, "db_record": db_insert_record}



@app.get("/")
async def root():
    return {"data":"Wellcome to MLOps API"}
