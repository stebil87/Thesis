import matplotlib.pyplot as plt
import seaborn as sns

def plot_spearman_correlation_matrix(df):
    eighth = len(df.columns) // 8
    slices = [df.columns[i * eighth:(i + 1) * eighth].tolist() for i in range(8)]  # Convert each slice to a list
    # Adjust the last slice to include any remaining columns
    slices[-1] = df.columns[(7 * eighth):].tolist() if len(df.columns) % 8 != 0 else slices[-1]
    # Ensure 'y' is in the last slice
    if 'y' not in slices[-1]:
        slices[-1].append('y')

    # Ensure 'y' is included in every part but only at the end of each list except the last part
    for i, slice_part in enumerate(slices[:-1]):
        if 'y' in slice_part:
            slice_part.remove('y')
        slice_part.append('y')  # Append 'y' to ensure it's always at the end for consistency

    # Plotting each segment's correlation matrix
    for i, segment in enumerate(slices):
        # Calculate correlation matrix for the current segment
        corr_matrix = df[segment].corr(method='spearman')
        plt.figure(figsize=(16, 14))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', center=0,
                    annot_kws={"size": 8}, cbar_kws={"shrink": .8})
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        plt.title(f'Spearman Correlation Matrix - Part {i+1}', fontsize=15)
        plt.show()

    # After plotting, calculate and print the top 20 unique correlations with 'y'
    corr_with_y = df.corr(method='spearman')['y'].drop('y', errors='ignore')  # Exclude self-correlation
    top_20_correlations = corr_with_y.abs().sort_values(ascending=False).head(20)
    print("Top 20 highest unique correlations with 'y':")
    print(top_20_correlations)

# Example usage, ensure you have a DataFrame `df` which includes the target variable 'y'
# plot_spearman_correlation_matrix(df)
