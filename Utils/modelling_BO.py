import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module='lightgbm')

# Function for Bayesian Optimization
def optimize_xgb(X_train, y_train):
    param_distributions = {
        'n_estimators': Integer(50, 200),
        'learning_rate': Real(0.01, 0.2, prior='log-uniform'),
        'max_depth': Integer(3, 7),
        'subsample': Real(0.6, 1.0, prior='uniform'),
        'colsample_bytree': Real(0.6, 1.0, prior='uniform')
    }

    model = XGBRegressor()
    bayes_search = BayesSearchCV(
        estimator=model,
        search_spaces=param_distributions,
        n_iter=10,
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    bayes_search.fit(X_train, y_train)
    return bayes_search.best_estimator_

def bayesian_optimization(datasets):
    results = {}
    predictions = {}

    dict_name = 'augmented_features_linear'
    print(f"Processing dictionary: {dict_name}")
    dataframes = list(datasets[dict_name].values())
    results[dict_name] = {}
    predictions[dict_name] = {}

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(dataframes):
        train_dfs = [dataframes[i] for i in train_index]
        test_df = dataframes[test_index[0]]

        X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])
        y_train = pd.concat([df['y'] for df in train_dfs])
        X_test = test_df.drop(columns='y', errors='ignore')
        y_test = test_df['y']

        print("Optimizing XGBoost...")
        best_model = optimize_xgb(X_train, y_train)
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        if 'XGBoost' not in results[dict_name]:
            results[dict_name]['XGBoost'] = []
        results[dict_name]['XGBoost'].append(mae)

        if 'XGBoost' not in predictions[dict_name]:
            predictions[dict_name]['XGBoost'] = []
        predictions[dict_name]['XGBoost'].append((y_test.values, y_pred))

    return results, predictions

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for model_name, maes in dict_results.items():
            avg_mae = np.mean(maes)
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")