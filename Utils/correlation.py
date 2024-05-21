import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_spearman_correlation_matrix(df):
    half = len(df.columns) // 2
    first_half = list(df.columns[:half]) + ['y']
    second_half = list(df.columns[half:-1]) + ['y']

    first_half = [col for col in first_half if col != 'y' or first_half.index(col) == len(first_half) - 1]
    second_half = [col for col in second_half if col != 'y' or second_half.index(col) == len(second_half) - 1]

    corr_matrix_first = df[first_half].corr(method='spearman')
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix_first, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Spearman Correlation Matrix - Part 1', fontsize=15)
    plt.show()
    
    corr_matrix_second = df[second_half].corr(method='spearman')
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix_second, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                annot_kws={"size": 8}, cbar_kws={"shrink": .8})
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.title('Spearman Correlation Matrix - Part 2', fontsize=15)
    plt.show()