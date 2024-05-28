import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor

def create_hankel_matrix(X, n_rows=None):
    hankel_data = []
    for column in X.columns:
        data = X[column].values
        if n_rows is None:
            n_rows = len(data) // 2
        hankel_matrix = np.array([data[i: i + n_rows] for i in range(len(data) - n_rows + 1)])
        hankel_data.append(hankel_matrix.flatten())
    return pd.DataFrame(hankel_data).transpose()

def perform_regression_and_cv(datasets, use_hankel=False):
    results = {}
    for name, df in datasets.items():
        print(f"Processing {name}...")
        
        X = df.drop('y', axis=1)
        y = df['y']
        
        if use_hankel:
            X = create_hankel_matrix(X)
        
        models = {
            'XGBoost': XGBRegressor(),
            'LightGBM': LGBMRegressor(),
            'AdaBoost': AdaBoostRegressor()
        }
        
        loo = LeaveOneOut()
        
        for model_name, model in models.items():
            mae_scores = []
            for train_index, test_index in loo.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                mae_scores.append(mean_absolute_error(y_test, y_pred))
            
            avg_mae = np.mean(mae_scores)
            if name not in results:
                results[name] = {}
            results[name][model_name] = avg_mae
            
            print(f"Model: {model_name}, MAE: {avg_mae:.4f}")
    
    return results

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for df_name, model_results in dict_results.items():
            print(f"  DataFrame: {df_name}")
            for model_name, mae in model_results.items():
                print(f"    {model_name}: MAE = {mae:.4f}")
