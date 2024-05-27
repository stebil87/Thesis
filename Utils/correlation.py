import pandas as pd

def compute_top_correlations(df, target_column='y', num_top=30):
    full_corr_matrix = df.corr(method='spearman')
    corr_with_y = full_corr_matrix[target_column].drop(target_column, errors='ignore').abs()
    top_columns = corr_with_y.sort_values(ascending=False).head(num_top).index.tolist()
    
    top_correlations = corr_with_y[top_columns].sort_values(ascending=False)
    return top_correlations

def get_top_correlations_for_all(dictionaries, target_column='y', num_top=30):
    top_correlations_dict = {}
    
    for dict_name, dict_data in dictionaries.items():
        top_correlations_dict[dict_name] = {}
        for df_name, df in dict_data.items():
            top_correlations = compute_top_correlations(df, target_column, num_top)
            top_correlations_dict[dict_name][df_name] = top_correlations
    
    return top_correlations_dict

def summarize_top_correlations(correlations_dict):
    summary = []
    
    for dict_name, dict_data in correlations_dict.items():
        for df_name, correlations in dict_data.items():
            for feature, correlation in correlations.items():
                summary.append({
                    'Dictionary': dict_name,
                    'DataFrame': df_name,
                    'Feature': feature,
                    'Correlation': correlation
                })
    
    summary_df = pd.DataFrame(summary)
    return summary_df

def display_top_correlations_summary(summary_df, num_features=10):
    print("Top Correlations Summary:")
    for dict_name in summary_df['Dictionary'].unique():
        print(f"\nDictionary: {dict_name}")
        dict_df = summary_df[summary_df['Dictionary'] == dict_name]
        for df_name in dict_df['DataFrame'].unique():
            print(f"  DataFrame: {df_name}")
            df_summary = dict_df[dict_df['DataFrame'] == df_name]
            print(df_summary.nlargest(num_features, 'Correlation')[['Feature', 'Correlation']])

