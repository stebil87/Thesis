import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import matplotlib.pyplot as plt

def bayesian_optimization(X_train, y_train, model, param_space):
    opt = BayesSearchCV(
        model,
        param_space,
        n_iter=10,  # Number of parameter settings that are sampled.
        cv=3,  # Cross-validation strategy
        n_jobs=-1,  # Use all available cores
        verbose=0
    )
    opt.fit(X_train, y_train)
    best_params = opt.best_params_
    return best_params

def BO_regression(dictionaries):
    models = {
        'XGBoost': XGBRegressor(objective='reg:squarederror'),
        'AdaBoost': AdaBoostRegressor(),
        'LightGBM': LGBMRegressor()
    }
    param_spaces = {
        'XGBoost': {
            'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(3, 10),
            'min_child_weight': Real(1, 10),
            'subsample': Real(0.5, 1.0, 'uniform'),
            'colsample_bytree': Real(0.5, 1.0, 'uniform'),
            'gamma': Real(0, 10, 'uniform')
        },
        'AdaBoost': {
            'n_estimators': Integer(50, 500),
            'learning_rate': Real(0.01, 1.0, 'log-uniform')
        },
        'LightGBM': {
            'learning_rate': Real(0.01, 1.0, 'log-uniform'),
            'n_estimators': Integer(100, 1000),
            'max_depth': Integer(3, 10),
            'num_leaves': Integer(20, 50),
            'subsample': Real(0.5, 1.0, 'uniform'),
            'colsample_bytree': Real(0.5, 1.0, 'uniform'),
            'min_child_samples': Integer(10, 30),
            'reg_alpha': Real(0.0, 1.0, 'uniform'),
            'reg_lambda': Real(0.0, 1.0, 'uniform')
        }
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
                # Perform Bayesian Optimization to find best hyperparameters
                best_params = bayesian_optimization(X_train, y_train, model, param_spaces[model_name])
                
                # Train model with best hyperparameters
                model.set_params(**best_params)
                model.fit(X_train, y_train)
                
                # Evaluate model
                predictions = model.predict(X_test)
                mae = mean_absolute_error(y_test, predictions)
                
                # Store results
                results[dict_name][model_name]['y_true'].append(y_test.values)
                results[dict_name][model_name]['y_pred'].append(predictions)
                results[dict_name][model_name]['mae'].append(mae)

    for dict_name, model_data in results.items():
        for model_name, data in model_data.items():
            # Print average MAE
            avg_mae = np.mean(data['mae'])
            print(f"Average MAE for {model_name} in {dict_name}: {avg_mae:.4f}")
            
            # Plot predictions
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
