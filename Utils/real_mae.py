from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics import mean_absolute_error

hardcoded_data = {
    25: ('2022.01.17 12:00', '2021-11-24 11:24:45.084000+00:00'),
    26: ('2022.01.14 23:30', '2021-11-24 12:00:54.253000+00:00'),
    27: ('2022.01.17 09:30', '2021-11-24 12:00:54.253000+00:00'),
    28: ('2022.01.06 03:30', '2021-11-24 12:00:54.253000+00:00'),
    29: ('2022.01.18 11:30', '2021-11-24 12:00:54.253000+00:00'),
    30: ('2022.02.01 01:00', '2021-11-24 12:00:54.253000+00:00'),
    31: ('2022.01.21 19:30', '2021-11-24 12:00:54.253000+00:00'),
    32: ('2022.01.22 01:30', '2021-11-24 12:00:54.253000+00:00'),
    1:  ('2022.01.31 23:30', '2021-11-24 12:00:59.258000+00:00'),
    2:  ('2022.02.08 05:30', '2021-11-24 12:00:59.258000+00:00'),
    3:  ('2022.01.07 23:30', '2021-11-24 12:00:59.258000+00:00'),
    4:  ('2022.01.30 05:30', '2021-11-24 11:24:45.084000+00:00'),
    5:  ('2022.01.04 23:30', '2021-11-24 12:00:59.258000+00:00'),
    6:  ('2022.01.26 11:30', '2021-11-24 12:00:59.258000+00:00'),
    7:  ('2022.01.24 23:30', '2021-11-24 12:00:59.258000+00:00'),
    8:  ('2022.01.17 23:30', '2021-11-24 12:00:59.258000+00:00'),
    17: ('2022.03.27 06:00', '2021-11-24 12:00:59.258000+00:00'),
    18: ('2022.03.30 19:00', '2021-11-24 23:00:02.801000+00:00'),
    19: ('2022.04.10 14:00', '2021-11-24 23:00:02.801000+00:00'),
    20: ('2022.03.09 17:00', '2021-11-24 23:00:02.801000+00:00'),
    21: ('2022.05.05 02:00', '2021-11-24 23:00:02.801000+00:00'),
    22: ('2022.04.04 20:00', '2021-11-24 23:00:02.801000+00:00'),
    23: ('2022.03.09 23:50', '2021-11-24 11:24:45.084000+00:00'),
    24: ('2022.04.08 14:00', '2021-11-24 23:00:02.801000+00:00'),
    9:  ('2022.04.17 10:00', '2021-11-24 23:00:02.801000+00:00'),
    10: ('2022.06.15 14:00', '2021-11-24 23:00:02.801000+00:00'),
    11: ('2022.05.22 23:50', '2021-11-24 11:24:45.084000+00:00'),
    12: ('2022.04.15 16:00', '2021-11-24 11:24:45.084000+00:00'),
    13: ('2022.03.18 08:30', '2021-11-24 11:24:45.084000+00:00'),
    14: ('2022.05.23 14:00', '2021-11-24 11:24:45.084000+00:00'),
    15: ('2022.05.17 02:00', '2021-11-24 11:24:45.084000+00:00'),
    16: ('2022.04.03 02:00', '2021-11-24 12:00:54.253000+00:00')
}

def compute_sprouting_mae(individual_predictions):
    # Parse the actual sprouting dates and start dates
    actual_dates = {df: datetime.strptime(sprout_day_time, "%Y.%m.%d %H:%M") for df, (sprout_day_time, _) in hardcoded_data.items()}
    start_dates = {df: datetime.fromisoformat(start_date) for df, (_, start_date) in hardcoded_data.items()}

    maes = {}
    for dict_name, model_predictions in individual_predictions.items():
        for model_name, predictions in model_predictions.items():
            predicted_sprouting_days = []
            for df_number, prediction in zip(actual_dates.keys(), predictions):
                start_date = start_dates[df_number]
                # Add the absolute value of the prediction to the start date
                predicted_date = start_date + timedelta(days=float(abs(prediction)))
                predicted_sprouting_days.append(predicted_date)

            # Calculate the mean predicted sprouting date
            mean_predicted_date = datetime.fromtimestamp(np.mean([d.timestamp() for d in predicted_sprouting_days]))

            # Calculate the error with the actual sprouting date
            actual_sprouting_date = actual_dates[df_number]
            error = abs((mean_predicted_date - actual_sprouting_date).total_seconds() / (60 * 60 * 24))
            
            if dict_name not in maes:
                maes[dict_name] = {}
            maes[dict_name][model_name] = error
    
    return maes
