import argparse
import os
import pickle
import numpy as np
import mlflow
from hyperopt import hp, space_eval
from hyperopt.pyll import scope
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import xgboost as xgb
from sklearn.metrics import mean_squared_error

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore')


HPO_EXPERIMENT_NAME = "electricity consumption v4"
EXPERIMENT_NAME = "xgboost-best-model"

os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5000/'
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:9000/'

mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.xgboost.autolog()


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

def mean_absolute_percentage_error(y_true, y_pred): 
    """Calculates MAPE given y_true and y_pred"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_valid, label=y_valid)
    
    with mlflow.start_run():
        
        booster = xgb.train(
        params=params,
        dtrain=train,
        num_boost_round=1000,
        evals=[(valid, 'validation')],
        early_stopping_rounds=50)
    
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_valid, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_valid, y_pred)

        
        mlflow.log_metric("valid_rmse", rmse)
        mlflow.log_metric("valid_mape", mape)


def run(data_path, log_top):
    
    print(f" TOP {log_top} model is traning...")    
    client = MlflowClient()

    # retrieve the top_n model runs and log the models to MLflow
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"]
    )
    # print(runs)
    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    # select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # register the best model
    run_id = best_run.info.run_id
    model_uri = f'runs:/{run_id}/model'
    mlflow.register_model(model_uri=model_uri, name='MWh-Consumption')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        default="./output",
        help="the location where data was saved."
    )
    parser.add_argument(
        "--top_n",
        default=5,
        type=int,
        help="the top 'top_n' models will be evaluated to decide which model to promote."
    )
    args = parser.parse_args()

    run(args.data_path, args.top_n)
