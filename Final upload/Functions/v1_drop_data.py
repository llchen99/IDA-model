def drop_data(threshold_column, threshold_row, df):
    '''With this function, data with less than threshold % of non-NaN values will be dropped from the dataset.
    Inputs:
        threshold_column: the threshold for columns to be dropped given as a value between 0 and 1
        threshold_row: the threshold for rows to be dropped given as a value between 0 and 1
        df:
    Returns:
        df_drop: dataframe with removed rows and columns. '''
    df_col = df.dropna(axis=1, thresh=threshold_column*list(df.shape)[0])
    df_drop = df_col.dropna(axis=0, thresh=threshold_row*list(df_col.shape)[1])
    return df_drop
