import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy import stats
from mlxtend.feature_selection import SequentialFeatureSelector

def find_sign_features(train_data, train_label, index_train, ordinal_keys, binary_keys, continuous_keys, sign_features_dfs):
    '''With this function, feature selection is done with univariate statistical testing. Dataframes with train data, train labels and train indices
    must be given as input in order to merge the train data and labels. Also lists of keys of ordinal features, binary features and continuous features
    must be given as input. A dataframe with significant features must be given as input and will be appended every fold. This appended list is returned
    and can be used for reporting. Also, a list of only the significant features in this fold is returned. This can be used for the creation of models
    with only significant features. Binary data is tested with Chi-square, ordinal data with multiple classes with a Mann-Whitney U test and continuous
    data with a Student's t-test. The p-values are corrected with a Holm-Bonferroni correction.'''
    # Merge data with labels again for statistics
    merge_data_train = train_data.merge(train_label, on=index_train, how='inner')
    # Create to dataframes for the different populations
    df_num_0 = merge_data_train.loc[merge_data_train['Label'] == 0.0]
    df_num_1 = merge_data_train.loc[merge_data_train['Label'] == 1.0]

    # Create dataframe to fill with p-values
    df_p = pd.DataFrame({'Features': ordinal_keys+binary_keys+continuous_keys})

    # Chi-square for binary data
    for key in binary_keys:
        _, p, _, _ = chi2_contingency(pd.crosstab(merge_data_train['Label'], merge_data_train[key]))   # Perform the Chi square test
        df_p.loc[df_p['Features'] == key, 'P-value'] = p   # Fill dataframe with p-values
        # Calculate the mean and std for the two populations and fill in dataframe
        mean_ID = np.round(df_num_1[key].mean(), decimals=2)
        std_ID = np.round(df_num_1[key].std(), decimals=2)
        mean_no_ID = np.round(df_num_0[key].mean(), decimals=2)
        std_no_ID = np.round(df_num_0[key].std(), decimals=2)
        df_p.loc[df_p['Features'] == key, 'Mean ± std ID'] = f'{mean_ID} ± {std_ID}'
        df_p.loc[df_p['Features'] == key, 'Mean ± std no ID'] = f'{mean_no_ID} ± {std_no_ID}'

    # Mann Whitney U test for multiple class ordinal data
    for key in ordinal_keys:
        _, p = mannwhitneyu(df_num_0[key], df_num_1[key])   # Perform the Mann Whitney U test
        df_p.loc[df_p['Features'] == key, 'P-value'] = p   # Fill dataframe with p-values
        # Calculate the mean and std for the two populations and fill in dataframe
        mean_ID = np.round(df_num_1[key].mean(), decimals=2)
        std_ID = np.round(df_num_1[key].std(), decimals=2)
        mean_no_ID = np.round(df_num_0[key].mean(), decimals=2)
        std_no_ID = np.round(df_num_0[key].std(), decimals=2)
        df_p.loc[df_p['Features'] == key, 'Mean ± std ID'] = f'{mean_ID} ± {std_ID}'
        df_p.loc[df_p['Features'] == key, 'Mean ± std no ID'] = f'{mean_no_ID} ± {std_no_ID}'

    # Student's t-test for continuous data
    for key in continuous_keys:
        _, p = stats.ttest_ind(df_num_0[key], df_num_1[key], nan_policy='omit')   # Perform the Student's t-test
        df_p.loc[df_p['Features'] == key, 'P-value'] = p    # Fill dataframe with p-values
        # Calculate the mean and std for the two populations and fill in dataframe
        mean_ID = np.round(df_num_1[key].mean(), decimals=2)
        std_ID = np.round(df_num_1[key].std(), decimals=2)
        mean_no_ID = np.round(df_num_0[key].mean(), decimals=2)
        std_no_ID = np.round(df_num_0[key].std(), decimals=2)
        df_p.loc[df_p['Features'] == key, 'Mean ± std ID'] = f'{mean_ID} ± {std_ID}'
        df_p.loc[df_p['Features'] == key, 'Mean ± std no ID'] = f'{mean_no_ID} ± {std_no_ID}'

    # Find significant p-values by Holm-Bonferroni:
    df_p_sorted = df_p.sort_values(by=['P-value'])    # Sort the values by p-values
    df_p_sorted['Rank'] = range(1, len(df_p_sorted)+1)    # Rank the features
    df_p_sorted['Significance level'] = 0.05/(len(df_p_sorted)+1-df_p_sorted['Rank'])    # Calculate the significance level per feature
    df_p_sorted['Significant'] = np.where(df_p_sorted['P-value'] < df_p_sorted['Significance level'], 'Yes', 'No')    # Find which features are significant

    # Create dataframe with significant features only and create table for visualisation
    df_p_sign = df_p_sorted.loc[df_p_sorted['Significant'] == 'Yes']
    df_p_for_table = df_p_sign.drop(['Rank'], axis=1)

    # Append the dataframe with significant features to a list for every fold. In this list, the dataframes for the 10 folds are stored.
    sign_features_dfs.append(df_p_for_table)

    # Create list of significant features that can be used for model creation
    sign = df_p_sign['Features'].tolist()
    return sign, sign_features_dfs

