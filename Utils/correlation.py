import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_spearman_correlation_matrix(df):
    corr_matrix = df.corr(method='spearman')
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Spearman Correlation Matrix', fontsize=15)
    plt.show()