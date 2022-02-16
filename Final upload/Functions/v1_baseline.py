import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt


def baseline(df_0_baseline, df_1_baseline):
    '''With this function, the age and gender of two dataframes will be statistically compared.
    Inputs:
        df_0_baseline: dataframe of subjects without ID including age and gender.
        df_1_baseline: dataframe of subject with ID including age and gender.
    Returns:
        df_characteristics: A dataframe with means and standard deviations of the two groups and a p-value indicating the difference
    between the two groups will be returned.'''
    # First calculate the means and stds of age in the two groups, rounded to two decimals
    mean_age_0 = np.round(df_0_baseline['Leeftijd'].mean(), decimals=2)
    std_age_0 = np.round(df_0_baseline['Leeftijd'].std(), decimals=2)
    mean_age_1 = np.round(df_1_baseline['Leeftijd'].mean(), decimals=2)
    std_age_1 = np.round(df_1_baseline['Leeftijd'].std(), decimals=2)
    # Next, find the percentage of females per group
    f_gender_0 = (df_0_baseline['Geslacht'].sum())/len(df_0_baseline)
    f_gender_1 = (df_1_baseline['Geslacht'].sum())/len(df_1_baseline)
    # Calculate the difference in gender with a Chi-square and the difference in age with a Student's t-test
    _, p_gender, _, _ = chi2_contingency(pd.crosstab(df_0_baseline['Geslacht'], df_1_baseline['Geslacht']))
    _, p_age = stats.ttest_ind(df_0_baseline['Leeftijd'], df_1_baseline['Leeftijd'])
    # Combine the calculated values into a dictionary, that is converted to a dataframe for visualisation.
    dict_table = {'Amount of patients': [f'N={len(df_1_baseline)}', f'N={len(df_0_baseline)}', ' '],
                  'Age': [f'{mean_age_1} ± {std_age_1}', f'{mean_age_0} ± {std_age_0}', np.round(p_age, decimals=2)],
                  'Gender': [f'{np.round(f_gender_1*100, decimals=0)}% females (N={np.round(f_gender_1*len(df_1_baseline), decimals=0)})',
                             f'{np.round(f_gender_0*100, decimals=0)}% females (N={np.round(f_gender_0*len(df_0_baseline), decimals=0)})', np.round(p_gender, decimals=2)]}
    df_characteristics = pd.DataFrame.from_dict(dict_table, orient='index', columns=['ID group', 'no ID group', 'P-value'])

    # Visualize age distribution
    fig1 = plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    ax = sns.boxplot(y='Leeftijd', data=df_1_baseline, color='skyblue').set_title('ID group')
    ax = sns.swarmplot(y='Leeftijd', data=df_1_baseline, color='k').set_title('ID group')

    plt.subplot(1, 2, 2)
    ax = sns.boxplot(y='Leeftijd', data=df_0_baseline, color='y').set_title('no ID group')
    ax = sns.swarmplot(y='Leeftijd', data=df_0_baseline, color='k').set_title('no ID group')

    # Visualize gender distribution
    fig, ax = plt.subplots()
    plt.subplot(1, 2, 1)
    rects1 = plt.hist(df_1_baseline['Geslacht'], 50)
    plt.xlabel('ID group')
    plt.ylabel('gender (%)')

    plt.subplot(1, 2, 2)
    rects0 = plt.hist(df_0_baseline['Geslacht'], 50)
    plt.xlabel('ID group')
    fig.suptitle('Distribution of gender')
    plt.show()
    return df_characteristics
