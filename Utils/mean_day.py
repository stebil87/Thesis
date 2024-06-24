import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
import lightgbm as lgb
from datetime import datetime, timedelta



def decimal_days_to_sessagesimal(decimal_days):
    days = int(decimal_days)
    fractional_day = decimal_days - days
    hours = int(fractional_day * 24)
    minutes = int((fractional_day * 24 - hours) * 60)
    seconds = int((((fractional_day * 24 - hours) * 60) - minutes) * 60)
    return days, hours, minutes, seconds

def day_predict(individual_predictions):
    mean_days = {}
    today = datetime.now()

    for dict_name, model_results in individual_predictions.items():
        mean_days[dict_name] = {}
        for model_name, predictions in model_results.items():
            mean_prediction = np.mean(predictions)
            if mean_prediction < 0:
                mean_prediction = abs(mean_prediction)
            days, hours, minutes, seconds = decimal_days_to_sessagesimal(mean_prediction)
            mean_day = today + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            mean_days[dict_name][model_name] = mean_day

    return mean_days

def prettyp(mean_days):
    for dict_name, model_results in mean_days.items():
        print(f"\nMean day of prediction for {dict_name}:")
        for model_name, mean_day in model_results.items():
            print(f"  {model_name}: Mean prediction date and time = {mean_day}")