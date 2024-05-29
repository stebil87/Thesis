import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

def create_hankel_matrix(X, n_rows=None):
    hankel_data = []
    for column in X.columns:
        data = X[column].values
        if n_rows is None:
            n_rows = len(data) // 2
        hankel_matrix = np.array([data[i: i + n_rows] for i in range(len(data) - n_rows + 1)])
        hankel_data.append(hankel_matrix.flatten())
    return pd.DataFrame(hankel_data).transpose()

def perform_regression_and_cv(dictionaries, use_hankel=False):
    results = {}
    loo = LeaveOneOut()

    for dict_name, datasets in dictionaries.items():
        print(f"Processing dictionary: {dict_name}")
        dataframes = list(datasets.values())
        results[dict_name] = {}

        for train_index, test_index in loo.split(dataframes):
            train_dfs = [dataframes[i] for i in train_index]
            test_df = dataframes[test_index[0]]

            X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])
            y_train = pd.concat([df['y'] for df in train_dfs])
            X_test = test_df.drop(columns='y', errors='ignore')
            y_test = test_df['y']

            if use_hankel:
                X_train = create_hankel_matrix(X_train)
                X_test = create_hankel_matrix(X_test)

            models = {
                'XGBoost': XGBRegressor(),
                'LightGBM': LGBMRegressor(),
                'AdaBoost': AdaBoostRegressor()
            }

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                if model_name not in results[dict_name]:
                    results[dict_name][model_name] = []
                results[dict_name][model_name].append(mae)

    return results

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for model_name, maes in dict_results.items():
            avg_mae = np.mean(maes)
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")


