import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def regression(dictionaries):
    models = {
        'XGBoost': XGBRegressor(objective='reg:squarederror'),
        'AdaBoost': AdaBoostRegressor(),
        'LightGBM': LGBMRegressor()
    }
    results = {key: {model: {'y_true': [], 'y_pred': [], 'mae': []} for model in models} for key in dictionaries}  
    
    for dict_name, df_dict in dictionaries.items():
        print(f"Processing {dict_name}...")
        if len(df_dict) < 2:
            print(f"Not enough data frames in {dict_name} for leave-one-out cross-validation.")
            continue

        for test_df_name, test_df in df_dict.items():
            train_dfs = [df for name, df in df_dict.items() if name != test_df_name]
            train_df = pd.concat(train_dfs)

            X_train = train_df.drop(columns='y')
            y_train = train_df['y']
            X_test = test_df.drop(columns='y')
            y_test = test_df['y']

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                results[dict_name][model_name]['y_true'].append(y_test.values)
                results[dict_name][model_name]['y_pred'].append(predictions)
                
                mae = mean_absolute_error(y_test, predictions)
                results[dict_name][model_name]['mae'].append(mae)

    for dict_name, model_data in results.items():
        for model_name, data in model_data.items():
            avg_mae = np.mean(data['mae'])
            print(f"Average MAE for {model_name} in {dict_name}: {avg_mae:.4f}")

            # Plotting predictions vs. ground truth
            plt.figure(figsize=(8, 6))
            all_y_true = np.concatenate(data['y_true'])
            all_y_pred = np.concatenate(data['y_pred'])
            plt.plot(all_y_true, label='Ground Truth', color='blue')
            plt.plot(all_y_pred, label='Predictions', linestyle='dashed', color='green')
            plt.title(f'{model_name} Predictions vs Ground Truth for {dict_name}')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            # Plotting box plot of prediction intervals
            plt.figure(figsize=(8, 6))
            plt.boxplot(data['y_pred'], labels=['XGBoost', 'AdaBoost', 'LightGBM'])
            plt.title(f'Prediction Intervals for {dict_name}')
            plt.xlabel('Models')
            plt.ylabel('Predicted Values')
            plt.grid(True)
            plt.tight_layout()
            plt.show()



"""
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

def regression(dictionaries):
    models = {
        'XGBoost': XGBRegressor(objective='reg:squarederror'),
        'AdaBoost': AdaBoostRegressor(),
        'LightGBM': LGBMRegressor()
    }
    results = {key: {model: {'y_true': [], 'y_pred': []} for model in models} for key in dictionaries}  
    
    for dict_name, df_dict in dictionaries.items():
        print(f"Processing {dict_name}...")
        if len(df_dict) < 2:
            print(f"Not enough data frames in {dict_name} for leave-one-out cross-validation.")
            continue

        for test_df_name, test_df in df_dict.items():
            train_dfs = [df for name, df in df_dict.items() if name != test_df_name]
            train_df = pd.concat(train_dfs)

            X_train = train_df.drop(columns='y')
            y_train = train_df['y']
            X_test = test_df.drop(columns='y')
            y_test = test_df['y']

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                results[dict_name][model_name]['y_true'].append(y_test.values)
                results[dict_name][model_name]['y_pred'].append(predictions)
                
                mae = mean_absolute_error(y_test, predictions)
                results[dict_name][model_name]['mae'] = mae

    for dict_name, model_data in results.items():
        for model_name, data in model_data.items():
            print(f"Average MAE for {model_name} in {dict_name}: {data['mae']:.4f}")

            plt.figure(figsize=(8, 6))
            all_y_true = np.concatenate(data['y_true'])
            all_y_pred = np.concatenate(data['y_pred'])
            plt.plot(all_y_true, label='Ground Truth', color='blue')
            plt.plot(all_y_pred, label='Predictions', linestyle='dashed', color='green')
            plt.title(f'{model_name} Predictions vs Ground Truth for {dict_name}')
            plt.xlabel('Index')
            plt.ylabel('Values')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
            """