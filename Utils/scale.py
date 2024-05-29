from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_features(df, target='y'):
    scaler = StandardScaler()
    X = df.drop(columns=[target], errors='ignore')
    y = df[target]
    X_scaled = scaler.fit_transform(X)
    df_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    df_scaled[target] = y
    return df_scaled
