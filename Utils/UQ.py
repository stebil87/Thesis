from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor

# Define the models
models = {
    'XGB': xgb.XGBRegressor(),
    'ADAB': AdaBoostRegressor(),
    'LGBM': LGBMRegressor()
}

def regression_UQ(models, dictionaries, sprouting_days, n_subsets=10, uncertainty_threshold=100.0):
    results = {}
    individual_maes = {}
    individual_predictions = {}
    delta_days = {}
    trained_models = {}

    for dict_name, df_dict in dictionaries.items():
        print(f"Evaluating {dict_name}...")
        results[dict_name] = {}
        individual_maes[dict_name] = {}
        individual_predictions[dict_name] = {}
        delta_days[dict_name] = {}
        trained_models[dict_name] = {}

        for model_name, model in models.items():
            print(f"  Using model {model_name}...")
            loo = LeaveOneOut()
            maes = []
            predictions = []
            predicted_dates = []

            df_names = list(df_dict.keys())

            for train_index, test_index in loo.split(df_names):
                train_df_names = [df_names[i] for i in train_index]
                test_df_name = df_names[test_index[0]]
                train_dfs = [df_dict[name] for name in train_df_names]
                test_df = df_dict[test_df_name]

                X_train_list = []
                y_train_list = []

                for train_df in train_dfs:
                    X_train_list.append(train_df.drop(columns=['y', 'timestamp'], errors='ignore').values)
                    y_train_list.append(train_df['y'].values)

                X_train_array = np.concatenate(X_train_list, axis=0)
                y_train_array = np.concatenate(y_train_list, axis=0)

                subset_models = []
                subset_size = len(X_train_array) // n_subsets
                indices = np.arange(len(X_train_array))
                np.random.shuffle(indices)

                for i in range(n_subsets):
                    subset_indices = indices[i * subset_size:(i + 1) * subset_size] if i < n_subsets - 1 else indices[i * subset_size:]
                    X_subset = X_train_array[subset_indices]
                    y_subset = y_train_array[subset_indices]
                    
                    subset_model = xgb.XGBRegressor()
                    subset_model.fit(X_subset, y_subset)
                    subset_models.append(subset_model)

                X_test = test_df.drop(columns=['y', 'timestamp'], errors='ignore').values
                y_test = test_df['y'].values

                subset_predictions = np.array([m.predict(X_test) for m in subset_models])
                mean_prediction = np.mean(subset_predictions, axis=0)
                prediction_interval = np.percentile(subset_predictions, [2.5, 97.5], axis=0)
                uncertainty = prediction_interval[1] - prediction_interval[0]
                
                print(f"  Uncertainty for {test_df_name} with model {model_name}: {uncertainty}")

                accepted_predictions = mean_prediction[uncertainty <= uncertainty_threshold]
                accepted_y_test = y_test[uncertainty <= uncertainty_threshold]

                if len(accepted_predictions) == 0:
                    print(f"  No accepted predictions for {test_df_name} with model {model_name}")
                    continue

                mae = mean_absolute_error(accepted_y_test, accepted_predictions)
                maes.append(mae)
                predictions.extend(accepted_predictions)

                for i, pred in enumerate(accepted_predictions):
                    pred_float = float(pred)
                    predicted_date = test_df['timestamp'].iloc[i] + timedelta(days=abs(pred_float))
                    predicted_dates.append(predicted_date)

            if predicted_dates:
                epoch = datetime.utcfromtimestamp(0)
                date_nums = [(date - epoch).total_seconds() for date in predicted_dates]
                avg_date_num = np.mean(date_nums)
                avg_predicted_date = epoch + timedelta(seconds=avg_date_num)
                
                ground_truth_date = sprouting_days.get(test_df_name, None)
                if ground_truth_date is not None:
                    delta = abs((avg_predicted_date - ground_truth_date).days)
                    delta_days[dict_name][model_name] = delta

            if len(maes) == 0:
                print(f"  No MAE calculated for {dict_name} with model {model_name}")
                results[dict_name][model_name] = float('nan')
            else:
                results[dict_name][model_name] = np.mean(maes)

            individual_maes[dict_name][model_name] = maes
            individual_predictions[dict_name][model_name] = predictions
            print(f"    Average MAE: {np.mean(maes):.4f}")

    return results, individual_maes, individual_predictions, delta_days, trained_models

def pretty_print(results, delta_days):
    for dict_name, model_results in results.items():
        print(f"\nAverage MAEs for {dict_name}:")
        for model_name, avg_mae in model_results.items():
            print(f"  {model_name}: Average MAE = {avg_mae:.4f}")
        
        print(f"\nAverage Delta Days for {dict_name}:")
        for model_name, avg_delta in delta_days.get(dict_name, {}).items():
            print(f"  {model_name}: Average Delta Days = {avg_delta:.4f}")

#################

results, individual_maes, individual_predictions, delta_days, trained_models = regression_UQ(models, dictionaries, sprouting_days)
pretty_print(results, delta_days)