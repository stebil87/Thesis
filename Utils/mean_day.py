import pandas as pd
from datetime import datetime, timedelta
import numpy as np

def decimal_days_to_sessagesimal(decimal_days):
    days = int(decimal_days)
    fractional_day = decimal_days - days
    hours = int(fractional_day * 24)
    minutes = int((fractional_day * 24 - hours) * 60)
    seconds = int((((fractional_day * 24 - hours) * 60) - minutes) * 60)
    return days, hours, minutes, seconds

def compute_mean_day_of_prediction(individual_maes):
    mean_days = {}
    today = datetime.now()

    for dict_name, model_results in individual_maes.items():
        mean_days[dict_name] = {}
        for model_name, maes in model_results.items():
            mean_mae = np.mean(maes)
            days, hours, minutes, seconds = decimal_days_to_sessagesimal(mean_mae)
            mean_day = today + timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
            mean_days[dict_name][model_name] = mean_day

    return mean_days

def pretty_print_mean_days(mean_days):
    for dict_name, model_results in mean_days.items():
        print(f"\nMean day of prediction for {dict_name}:")
        for model_name, mean_day in model_results.items():
            print(f"  {model_name}: Mean prediction date and time = {mean_day}")