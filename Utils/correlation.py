
import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_correlations_with_y(df, num_top=30):
    full_corr_matrix = df.corr(method='spearman')
    corr_with_y = full_corr_matrix['y'].drop('y', errors='ignore').abs()
    top_columns = corr_with_y.sort_values(ascending=False).head(num_top).index.tolist()
    
    if 'y' not in top_columns:
        top_columns.append('y')
    
    reduced_corr_matrix = full_corr_matrix.loc[top_columns, top_columns]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(reduced_corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Top 30 Absolute Spearman Correlations with y', fontsize=15)
    plt.show()
