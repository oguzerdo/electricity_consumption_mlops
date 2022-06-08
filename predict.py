import os
import mlflow
import pandas as pd
import numpy as np
# Tell where is the tracking server and artifact server
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'


RUN_ID = "12d6e85e9b8d44cbb27a10779636dabe"

logged_model = f"s3://vbo-mlflow-bucket/10/{RUN_ID}/artifacts/model"
model = mlflow.pyfunc.load_model(logged_model)


pred_date = "2022-05-31 18:00"
data_dict = {"Date": pred_date,
             "MWh": None}


from preprocess_data import create_features


dataframe = pd.DataFrame(data_dict, index=[0])
dataframe["Date"] = pd.to_datetime(dataframe["Date"])
dataframe = dataframe.set_index("Date",drop=True)

pred_data = create_features(dataframe)
pred_data.drop("MWh", inplace=True, axis=1)

import pickle
def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)



model.predict(pred_data)


X_train, y_train = load_pickle(os.path.join("output", "train.pkl"))
X_valid, y_valid = load_pickle(os.path.join('output', "valid.pkl"))

X_valid.head()

X_valid.index.max()