import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def iscollinear(df):
    X = df.drop(columns=['y']) 
    X['Intercept'] = 1
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    vif_data = vif_data[vif_data['Feature'] != 'Intercept']
    
    return vif_data