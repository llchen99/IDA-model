def drop_data(threshold_column, threshold_row, df):
    '''With this function, data with less than threshold % of non-NaN values will be dropped from the dataset.
    The thresholds for dropping columns and rows must be specified as a value between 0 and 1 and the input must be a dataframe.
    The dataframe with removed rows and columns is returned.'''
    df_col = df.dropna(axis=1, thresh=threshold_column*list(df.shape)[0])
    df_drop = df_col.dropna(axis=0, thresh=threshold_row*list(df_col.shape)[1])
    return df_drop