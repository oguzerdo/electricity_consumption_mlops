import os
import mlflow
import pandas as pd
from scripts.preprocess_data import create_features

# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'


RUN_ID = "12d6e85e9b8d44cbb27a10779636dabe"

logged_model = f"s3://vbo-mlflow-bucket/10/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)

def prediction(model_obj, pred_date = "2022-05-31 18:00"):

    data_dict = {"Date": pred_date,
                "MWh": None}

    dataframe = pd.DataFrame(data_dict, index=[0])
    dataframe["Date"] = pd.to_datetime(dataframe["Date"])
    dataframe = dataframe.set_index("Date",drop=True)

    pred_data = create_features(dataframe)
    pred_data.drop("MWh", inplace=True, axis=1)

    prediction = model_obj.predict(pred_data)
    
    return prediction[0]