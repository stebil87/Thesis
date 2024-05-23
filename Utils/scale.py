from sklearn.preprocessing import RobustScaler
import pandas as pd

def scale_features(df, target_column):
  
    X = df.drop(target_column, axis=1) 
    y = df[target_column]              
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    df_scaled[target_column] = y 
    return df_scaled
