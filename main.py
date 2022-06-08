from fastapi import FastAPI, Depends, Request
from schemas import cons
import models
from database import engine, get_db
from sqlalchemy.orm import Session
import os
import mlflow
import pandas as pd
from preprocess_data import create_features

def get_model(RUN_ID = "12d6e85e9b8d44cbb27a10779636dabe"):
    
    os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'
    logged_model = f"s3://vbo-mlflow-bucket/10/{RUN_ID}/artifacts/model"
    model = mlflow.pyfunc.load_model(logged_model)

    return model


app = FastAPI()

# Creates all the tables defined in models module
models.Base.metadata.create_all(bind=engine)


def insert_mwh(request, prediction, client_ip, db):
    new_mwh = models.Consumption(
        date_=request["date_"],
        prediction=prediction,
        client_ip=client_ip
    )

    db.add(new_mwh)
    db.commit()
    db.refresh(new_mwh)
    return new_mwh



def prediction_mwh(model_obj, request):

    data_dict = {"Date": request["date_"],
                "MWh": None}

    dataframe = pd.DataFrame(data_dict, index=[0])
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe = dataframe.set_index("Date",drop=True)

    pred_data = create_features(dataframe)
    pred_data.drop("MWh", inplace=True, axis=1)

    prediction = model_obj.predict(pred_data)
    
    return float(prediction[0])



# prediction(get_model(), {"date_":"2022-06-01"})


# Iris Prediction endpoint
@app.post("/prediction")
def predict_iris(request: cons, fastapi_req: Request,  db: Session = Depends(get_db)):
    prediction = prediction_mwh(get_model(), request.dict())
    db_insert_record = insert_mwh(request=request.dict(), prediction=prediction,
                                          client_ip=fastapi_req.client.host,
                                          db=db)
    return {"prediction": prediction, "db_record": db_insert_record}


@app.get("/")
async def root():
    return {"data":"Wellcome to MLOps API"}
