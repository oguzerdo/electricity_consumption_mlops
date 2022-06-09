import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import holidays
import warnings
warnings.filterwarnings("ignore")


def create_features(dataframe):
    """
    Creates time series features from datetime index
    """
    dataframe['date'] = dataframe.index
    dataframe['hour'] = dataframe['date'].dt.hour
    dataframe['dayofweek'] = dataframe['date'].dt.dayofweek
    dataframe['quarter'] = dataframe['date'].dt.quarter
    dataframe['month'] = dataframe['date'].dt.month
    dataframe['year'] = dataframe['date'].dt.year
    dataframe['dayofyear'] = dataframe['date'].dt.dayofyear
    dataframe['dayofmonth'] = dataframe['date'].dt.day
    dataframe['weekofyear'] = dataframe['date'].dt.weekofyear
    dataframe['6_hrs_lag'] = dataframe['MWh'].shift(6)
    dataframe['12_hrs_lag'] = dataframe['MWh'].shift(12)
    dataframe['24_hrs_lag'] = dataframe['MWh'].shift(24)
    dataframe['6_hrs_mean'] = dataframe['MWh'].rolling(window = 6).mean()
    dataframe['12_hrs_mean'] = dataframe['MWh'].rolling(window = 12).mean()
    dataframe['24_hrs_mean'] = dataframe['MWh'].rolling(window = 24).mean()
    dataframe['6_hrs_std'] = dataframe['MWh'].rolling(window = 6).std()
    dataframe['12_hrs_std'] = dataframe['MWh'].rolling(window = 12).std()
    dataframe['24_hrs_std'] = dataframe['MWh'].rolling(window = 24).std()
    dataframe['6_hrs_max'] = dataframe['MWh'].rolling(window = 6).max()
    dataframe['12_hrs_max'] = dataframe['MWh'].rolling(window = 12).max()
    dataframe['24_hrs_max'] = dataframe['MWh'].rolling(window = 24).max()
    dataframe['6_hrs_min'] = dataframe['MWh'].rolling(window = 6).min()
    dataframe['12_hrs_min'] = dataframe['MWh'].rolling(window = 12).min()
    dataframe['24_hrs_min'] = dataframe['MWh'].rolling(window = 24).min()
    
    
    # Get all Turkish holiday dates from 2016
    holiday_list = []
    for holiday in holidays.Turkey(years=[2016, 2017, 2018, 2019, 2020, 2021, 2022]).items():
        holiday_list.append(holiday)

    holidays_df = pd.DataFrame(holiday_list, columns=["date", "holiday"])

    dataframe["Date"]= dataframe.index.date
    dataframe.loc[dataframe['Date'].isin(holidays_df["date"]), "holiday"] = 1
    dataframe.holiday.fillna(0, inplace=True)
    dataframe["holiday"] = dataframe["holiday"].astype("int")
    dataframe.drop("Date", inplace=True, axis=1)
    
    
    return dataframe


def dump_pickle(obj, filename):
    with open(filename, "wb") as f_out:
        return pickle.dump(obj, f_out)


def read_dataframe(filename: str):
    df = pd.read_csv(filename, encoding = "cp1252", decimal=',', 
                    thousands='.', parse_dates=['Tarih'],
                    infer_datetime_format="%d.%m.%Y")
    
    df.columns = ["Date", "Hour", "MWh"]

    # Create datetime
    df["Datetime"] = df.apply(lambda x: str(x["Date"].date()) + " " + x["Hour"], axis=1)
    df["Datetime"] = pd.to_datetime(df["Datetime"], format='%Y-%m-%d %H:%M')
    df = df[["Datetime", "MWh"]].copy()
    df = df.set_index("Datetime",drop=True)
    
    return df


def preprocess(dataframe: pd.DataFrame):
    df = create_features(dataframe)
    
    return df


def run(raw_data_path: str, dest_path: str):
    # load data files
    df = read_dataframe(os.path.join(raw_data_path, "data-01011992-04062022.csv"))
    
    df = preprocess(df)
    
    # Split data
    split_date: str='2022-01-01'
    
    train = df.loc[df.index < split_date].copy() # 2015 - 2022
    val = df.loc[df.index >= split_date].copy() # 2022.01 - 2022.06

    cols = [col for col in train.columns if col not in ["date", "MWh"]]

    y_train = train['MWh']
    X_train = train[cols]

    y_valid = val['MWh']
    X_valid = val[cols]
    

    # create dest_path folder unless it already exists
    os.makedirs(dest_path, exist_ok=True)

    # save datasets
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_valid, y_valid), os.path.join(dest_path, "valid.pkl"))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        default="./data",
        help="the location where the raw NYC taxi trip data was saved"
    )
    parser.add_argument(
        "--dest_path",
        default="./output",
        help="the location where the resulting files will be saved."
    )
    parser.add_argument(
        "--split_date",
        default="2022-01-01",
        help="split date for train and validation data"
    )
    
    args = parser.parse_args()

    run(args.raw_data_path, args.dest_path)


