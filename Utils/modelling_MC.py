import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='xgboost')

class MCDropoutXGBRegressor(XGBRegressor):
    def __init__(self, dropout_rate=0.1, n_estimators=100, random_state=None, **kwargs):
        super().__init__(n_estimators=n_estimators, random_state=random_state, **kwargs)
        self.dropout_rate = dropout_rate
        self.random_state = random_state

    def fit(self, X, y, **kwargs):
        self._train_data = X
        self._train_labels = y
        super().fit(X, y, **kwargs)

    def predict(self, X, n_iter=100):
        predictions = []
        rng = np.random.default_rng(self.random_state)
        for _ in range(n_iter):
            dropout_mask = rng.binomial(1, 1 - self.dropout_rate, size=X.shape)
            X_dropped = X * dropout_mask
            predictions.append(super().predict(X_dropped))
        return np.mean(predictions, axis=0), np.percentile(predictions, 2.5, axis=0), np.percentile(predictions, 97.5, axis=0)

def mc_dropout(dictionaries, dropout_rate=0.1, n_iter=100, uncertainty_threshold=0.1, random_state=None):
    results = {}
    predictions = {}
    loo = LeaveOneOut()

    for dict_name, datasets in dictionaries.items():
        print(f"Processing dictionary: {dict_name}")
        dataframes = list(datasets.values())
        results[dict_name] = {}
        predictions[dict_name] = {}

        for train_index, test_index in loo.split(dataframes):
            train_dfs = [dataframes[i] for i in train_index]
            test_df = dataframes[test_index[0]]

            X_train = pd.concat([df.drop(columns='y', errors='ignore') for df in train_dfs])
            y_train = pd.concat([df['y'] for df in train_dfs])
            X_test = test_df.drop(columns='y', errors='ignore')
            y_test = test_df['y']

            model = MCDropoutXGBRegressor(dropout_rate=dropout_rate, random_state=random_state)
            model.fit(X_train, y_train)
            mean_prediction, lower_bound, upper_bound = model.predict(X_test, n_iter=n_iter)

            uncertainty = upper_bound - lower_bound
            accepted_prediction = mean_prediction.copy()
            accepted_prediction[uncertainty > uncertainty_threshold] = np.nan  

            accepted_indices = ~np.isnan(accepted_prediction)
            if np.any(accepted_indices):
                mae = mean_absolute_error(y_test[accepted_indices], accepted_prediction[accepted_indices])
            else:
                mae = np.nan

            if 'XGBoost' not in results[dict_name]:
                results[dict_name]['XGBoost'] = []
            results[dict_name]['XGBoost'].append(mae)
            
            if 'XGBoost' not in predictions[dict_name]:
                predictions[dict_name]['XGBoost'] = []
            predictions[dict_name]['XGBoost'].append((y_test.values, mean_prediction, lower_bound, upper_bound))

    return results, predictions

def print_results(results, description):
    print(f"\n{description} Results:")
    for dict_name, dict_results in results.items():
        print(f"\nDictionary: {dict_name}")
        for model_name, maes in dict_results.items():
            avg_mae = np.nanmean(maes) 
            print(f"  Model: {model_name}, MAE: {avg_mae:.4f}")