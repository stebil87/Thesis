def drop_columns(df, columns_to_drop):
    return df.drop(columns=columns_to_drop, errors='ignore')