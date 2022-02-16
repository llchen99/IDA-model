import numpy as np
import pandas as pd
from sklearn import preprocessing


def scale_data(data_train, data_test, continuous_keys, ordinal_keys):
    '''With this function, data is scaled between 0 and 1. Scaling is done differently for different types of variables.
    Continuous data is scaled between 0 and 1 with a StandardScaler. Scaling ordinal keys with multiple classes is done with conditional probabilities.
    Inputs:
        data_train: dataframe of training data.
        data_test: dataframe of test data.
        continuous_keys: list of continuous keys to be scaled.
        ordinal_keys: list of ordinal keys to be scaled.
    Returns:
        data_train: dataframe of scaled training data.
        data_test: dataframe of scaled test data.
    '''

    # First scale the continuous features with a MinMaxScaler:
    scaler = preprocessing.StandardScaler()  # Define the scaler
    df_for_scaler_train = data_train[continuous_keys].copy()  # Create a copy dataframe with only the continuous keys of the train data
    df_for_scaler_test = data_test[continuous_keys].copy()   # Create a copy dataframe with only the continuous keys of the test data

    # Scale the continuous features of the train and test data
    scale_train = scaler.fit_transform(df_for_scaler_train)
    scale_test = scaler.transform(df_for_scaler_test)

    # Replace the columns of train and test data by the scaled values
    data_train.loc[:, (continuous_keys)] = scale_train
    data_test.loc[:, (continuous_keys)] = scale_test

    # Next, scale the ordinal features by replacing them with conditional probabilities:
    prob = []   # Create an empty list that will be filled by the calculated probabilities
    prob_train = data_train[ordinal_keys].copy()   # Create a copy dataframe with only the ordinal keys from the train data
    prob_test = data_test[ordinal_keys].copy()   # Create a copy dataframe with only the ordinal keys from the test data

    for ordinal in ordinal_keys:   # Loop over all the ordinal keys
        if int(prob_train[ordinal].max()) > 0:   # Only if the max of the ordinal column is larger than 0
            for i in range(1, int(prob_train[ordinal].max()+1)):   # Loop over range from 1 to the maximum value in the ordinal column
                prob_train['Ordinal value'] = np.where(prob_train[ordinal] > 0, 1, 0)   # Select only patients with >0 for this ordinal value
                prob_train[f"Amount = {i}"] = np.where(prob_train[ordinal] == i, 1, 0)   # Select only patients with i for this ordinal value
                prob_train['count'] = 1   # A help for developing a pivot table
                prob_df = prob_train[['Ordinal value', f"Amount = {i}", 'count']]   # Create a dataframe with only the columns necessary for creating a pivot table
                table = pd.pivot_table(prob_df, values='count', index=[f"Amount = {i}"], columns=['Ordinal value'], aggfunc=np.size, fill_value=0)   # Create a pivot table
                if table.size == 4:   # If the pivot table exists of 4 values, this means that i exists in the column
                    # Calculate the conditional probability with the following three rows
                    P_B = (table.values[0, 1]+table.values[1, 1])/(sum(sum(table.values)))
                    P_A_B = table.values[1, 1]/(sum(sum(table.values)))
                    prob.append(P_A_B / P_B)
                    prob_train[ordinal].replace({i: prob[-1]}, inplace=True)   # Replace i in the train column by the last added probability
                    prob_test[ordinal].replace({i: prob[-1]}, inplace=True)   # Replace i in the train column by the last added probability
                else:
                    continue

    prob_train_final = prob_train[prob_test.keys()]   # Remove the keys that were added for the pivot table
    # Replace the columns of train and test data by the scaled values
    data_train.loc[:, (ordinal_keys)] = prob_train_final[ordinal_keys]
    data_test.loc[:, (ordinal_keys)] = prob_test[ordinal_keys]
    return data_train, data_test
