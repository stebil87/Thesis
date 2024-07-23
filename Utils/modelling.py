from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sprouting_days = {
    "df25": datetime(2022, 1, 17, 12, 0),
    "df26": datetime(2022, 1, 14, 23, 30),
    "df27": datetime(2022, 1, 17, 9, 30),
    "df28": datetime(2022, 1, 6, 3, 30),
    "df29": datetime(2022, 1, 18, 11, 30),
    "df30": datetime(2022, 2, 1, 1, 0),
    "df31": datetime(2022, 1, 21, 19, 30),
    "df32": datetime(2022, 1, 22, 1, 30),
    "df1":  datetime(2022, 1, 31, 23, 30),
    "df2":  datetime(2022, 2, 8, 5, 30),
    "df3":  datetime(2022, 1, 7, 23, 30),
    "df4":  datetime(2022, 1, 30, 5, 30),
    "df5":  datetime(2022, 1, 4, 23, 30),
    "df6":  datetime(2022, 1, 26, 11, 30),
    "df7":  datetime(2022, 1, 24, 23, 30),
    "df8":  datetime(2022, 1, 17, 23, 30),
    "df17": datetime(2022, 3, 27, 6, 0),
    "df18": datetime(2022, 3, 30, 19, 0),
    "df19": datetime(2022, 4, 10, 14, 0),
    "df20": datetime(2022, 3, 9, 17, 0),
    "df21": datetime(2022, 5, 5, 2, 0),
    "df22": datetime(2022, 4, 4, 20, 0),
    "df23": datetime(2022, 3, 9, 23, 50),
    "df24": datetime(2022, 4, 8, 14, 0),
    "df9":  datetime(2022, 4, 17, 10, 0),
    "df10": datetime(2022, 6, 15, 14, 0),
    "df11": datetime(2022, 5, 22, 23, 50),
    "df12": datetime(2022, 4, 15, 16, 0),
    "df13": datetime(2022, 3, 18, 8, 30),
    "df14": datetime(2022, 5, 23, 14, 0),
    "df15": datetime(2022, 5, 17, 2, 0),
    "df16": datetime(2022, 4, 3, 2, 0)
}


def regression(models, dictionaries, sprouting_days):
    results = {}
    individual_maes = {}
    individual_predictions = {}
    delta_days = {}

    for dict_name, df_dict in dictionaries.items():
        print(f"Evaluating {dict_name}...")
        results[dict_name] = {}
        individual_maes[dict_name] = {}
        individual_predictions[dict_name] = {}
        delta_days[dict_name] = {}

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

                X_test = test_df.drop(columns=['y', 'timestamp'], errors='ignore').values
                y_test = test_df['y'].values

                model.fit(X_train_array, y_train_array)
                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                maes.append(mae)
                predictions.extend(y_pred)
                
                for i, pred in enumerate(y_pred):
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

            results[dict_name][model_name] = np.mean(maes)
            individual_maes[dict_name][model_name] = maes
            individual_predictions[dict_name][model_name] = predictions
            print(f"    Average MAE: {np.mean(maes):.4f}")

    return results, individual_maes, individual_predictions, delta_days

def pretty_print(results, delta_days):
    for dict_name, model_results in results.items():
        print(f"\nAverage MAEs for {dict_name}:")
        for model_name, avg_mae in model_results.items():
            print(f"  {model_name}: Average MAE = {avg_mae:.4f}")
        
        print(f"\nAverage Delta Days for {dict_name}:")
        for model_name, avg_delta in delta_days.get(dict_name, {}).items():
            print(f"  {model_name}: Average Delta Days = {avg_delta:.4f}")

