import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
import lightgbm as lgb
from datetime import datetime, timedelta

def regression(models, dictionaries):
    results = {}
    individual_maes = {}
    individual_predictions = {}

    for dict_name, df_dict in dictionaries.items():
        print(f"Evaluating {dict_name}...")
        results[dict_name] = {}
        individual_maes[dict_name] = {}
        individual_predictions[dict_name] = {}

        for model_name, model in models.items():
            print(f"  Using model {model_name}...")
            loo = LeaveOneOut()
            maes = []
            predictions = []

            for test_df_name in df_dict.keys():
                test_df = df_dict[test_df_name]
                train_dfs = [df for name, df in df_dict.items() if name != test_df_name]
                train_df = pd.concat(train_dfs)

                X_train, y_train = train_df.drop(columns=['y']), train_df['y']
                X_test, y_test = test_df.drop(columns=['y']), test_df['y']

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                maes.append(mae)
                predictions.extend(y_pred)

            results[dict_name][model_name] = np.mean(maes)
            individual_maes[dict_name][model_name] = maes
            individual_predictions[dict_name][model_name] = predictions
            print(f"    Average MAE: {np.mean(maes):.4f}")

    return results, individual_maes, individual_predictions


def pretty_print(results):
    for dict_name, model_results in results.items():
        print(f"\nAverage MAEs for {dict_name}:")
        for model_name, avg_mae in model_results.items():
            print(f"  {model_name}: Average MAE = {avg_mae:.4f}")




"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
import lightgbm as lgb

def regression(models, dictionaries):
    results = {}
    individual_maes = {}  
    
    for dict_name, df_dict in dictionaries.items():
        print(f"Evaluating {dict_name}...")
        results[dict_name] = {}
        individual_maes[dict_name] = {}
        
        for model_name, model in models.items():
            print(f"  Using model {model_name}...")
            loo = LeaveOneOut()
            maes = []

            for test_df_name in df_dict.keys():
                test_df = df_dict[test_df_name]
                train_dfs = [df for name, df in df_dict.items() if name != test_df_name]
                train_df = pd.concat(train_dfs)

                X_train, y_train = train_df.drop(columns=['y']), train_df['y']
                X_test, y_test = test_df.drop(columns=['y']), test_df['y']

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, y_pred)
                maes.append(mae)

            results[dict_name][model_name] = np.mean(maes)
            individual_maes[dict_name][model_name] = maes
            print(f"    Average MAE: {np.mean(maes):.4f}")
    
    return results, individual_maes


def pretty_print(results):
    for dict_name, model_results in results.items():
        print(f"\nAverage MAEs for {dict_name}:")
        for model_name, avg_mae in model_results.items():
            print(f"  {model_name}: Average MAE = {avg_mae:.4f}")
            """