from statistics import mean
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer

def impute_data(data_train, data_test, df_decimal):
    '''With this function, missing values are imputed with a 3-Nearest Neighbors imputer with a weight depending on distance.
    The imputer is fit on the train set and applied to the test set. The imputed values are rounded differently for every column.
    Two dataframes with train and test data must be given as input. The amount of decimals per feature must also be given as input in a separate dataframe.
    Two dataframes with train and test data with imputed (and rounded) values are returned.'''
    # Find the feature names in the data
    columns_data = list(data_train.columns)

    # Imputation of NaN's with KNN
    imputer = KNNImputer(n_neighbors=3, weights="distance")
    impute_train = imputer.fit_transform(data_train)
    impute_test = imputer.transform(data_test)

    # Create dataframes of train and test data with imputed values
    df_train = pd.DataFrame(impute_train, columns=columns_data)
    df_test = pd.DataFrame(impute_test, columns=columns_data)

    # Round the imputed values differently for every column
    for column in columns_data:   # Loop over the different features
        df_train.loc[:, column] = np.round(df_train[column], decimals=int(df_decimal[column].to_string(index=False)))   # Convert to string to remove the index of the dataframe. Convert to int because the amount of decimals is an integer.
        df_test.loc[:, column] = np.round(df_test[column], decimals=int(df_decimal[column].to_string(index=False)))
    return df_train, df_test