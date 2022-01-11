# LOG OF CHANGES
# Gewicht verwijderd.
# Geprobeerd om het imputeren niet van BMI te doen, niet gelukt, later terugkomen. Error bij de t-test. Misschien later BMI in een aparte dataframe doen en daarvan de NaNs te droppen? Teveel moeite? 
# Implementatie wrapper feature selection.

import os
import sys
clear = lambda: os.system('cls')  # On Windows System
clear()
sys.path.insert(0, r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Stage 2\IDA-model-main\IDA-model\Functions')

# Import the right data packages
import pandas as pd
import numpy as np
import seaborn as sns
from v1_baseline import baseline
from v1_drop_data import drop_data
from v1_feature_selection import find_sign_features
from v1_impute_data import impute_data
from v1_load_data import load_data
from v1_mean_ROC_curves import mean_ROC_curves
from v1_pipeline_model import pipeline_model
from v1_scale_data import scale_data
from v1_stratify import stratify
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import model_selection
from sklearn import metrics
from sklearn.inspection import permutation_importance
from sklearn.svm import SVC
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import mannwhitneyu
from scipy import stats
from statistics import mean
from statistics import stdev
from mlxtend.feature_selection import SequentialFeatureSelector as sfs

# Load and merge data. Define paths and columns wanted from Excel files
path_data = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/v7_dataset.xlsx'
columns_data = "A:BBB"
path_labels = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/IDA_aangevuld.xls'
columns_labels = "A:B"
path_phecodes = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/phecodes_filled_binary.xlsx'
columns_phecodes = "A:R"
path_specialisms = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/Letters_DBC_combined.xlsx'
columns_specialisms = "A:BW"
path_decimals = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/afronden_features.xlsx'
columns_decimals = 'A:HI'
path_baseline = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/baseline.xlsx'
columns_baseline = 'A:C'
path_extra = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/v2_extra.xlsx'
columns_extra = 'A:D'
path_brieven = 'C:/Users/linda/Dropbox/TM/Stagedocumenten/Stage 2/IDA-model-main/model/v2_brieven.xlsx'
columns_brieven = 'A:BR'

# df data contains most of the features needed to build the model
df_data = load_data(path_data, columns_data)
# df labels contains the label ID/ no ID
df_labels = load_data(path_labels, columns_labels)
# df phecodes contains the data about the diagnoses a patient has, categorized into groups
df_phecodes = load_data(path_phecodes, columns_phecodes)
# df spec contains the data about what specialisms a patient visited
df_spec = load_data(path_specialisms, columns_specialisms)
# df decimal contains the amount of decimals for rounding the different features
df_decimal = load_data(path_decimals, columns_decimals)
# df baseline contains baseline characteristics age and gender
df_baseline = load_data(path_baseline, columns_baseline)
# df_extra contains extra characteristics like BMI and opnames
df_extra = load_data(path_extra, columns_extra)
# df_brieven contains the letters sent from specialisms
df_brieven = load_data(path_brieven,columns_brieven)

# Merge dataframes
df_hix_spec = df_data.merge(df_spec, on='Pt_no', how='outer')
df_hix_phecodes = df_hix_spec.merge(df_phecodes, on='Pt_no', how='outer')
# In df hix, all features are merged inside one dataframe. The specialisms, phecodes and labels are added
df_hix = df_hix_phecodes.merge(df_labels, on='Pt_no', how='inner')
# Now add all the extra features
df_ex = df_hix.merge(df_extra, on='Pt_no',how = 'inner')
# Now add the letters from the specialisms
df_all = df_ex.merge(df_brieven, on = 'Pt_no', how = 'inner')
print('Number of columns before dropped columns: ' + str(len(df_all.columns)))

# Defining thresholds for dropping rows and columns with missing data (threshold of amount of non-NA values required)
threshold_column = 0.7
threshold_row = 0.6

# Drop columns and rows with too many NaN's
df_dropped = drop_data(threshold_column, threshold_row, df_all)

print('Number of columns after dropped columns: ' + str(len(df_dropped.columns)))
print(f'Shape of df_dropped = {df_dropped.shape}')

# Balance set by picking random samples from no ID group # hierrr
df_ID_1 = df_dropped.loc[df_dropped['Label'] == 1.0]
df_ID_0_all = df_dropped.loc[df_dropped['Label'] == 0.0]

# To perform stratified random sampling, the gender of the ID group is checked
# Check baseline characteristics of subset
# Merge the dataframes of ID and no ID with the baseline characteristics in df baseline (age and gender)
# Exploratory data analysis
df_0_all_baseline = df_ID_0_all.merge(df_baseline, on='Pt_no', how='inner')
df_1_baseline = df_ID_1.merge(df_baseline, on='Pt_no', how='inner')
#characteristics = baseline(df_0_all_baseline, df_1_baseline)

# Stratify the data
df_control = stratify(df_0_all_baseline, df_1_baseline)

# Nou moeten de dataframes van no ID en ID weer bij elkaar 
df_comb = pd.concat([df_1_baseline,df_control])
df_comb = df_comb.drop(['Leeftijd', 'Geslacht'], axis = 1 )

# Defining empty lists needed later
tprs_RF_all = []
aucs_RF_all = []
auc_RF_all = []
spec_RF_all = []
sens_RF_all = []
accuracy_RF_all = []
tprs_RF_fin = []
aucs_RF_fin = []
auc_RF_fin = []
spec_RF_fin = [] 
sens_RF_fin = []
accuracy_RF_fin = []
tprs_SVM_fin = []
aucs_SVM_fin = [] 
auc_SVM_fin = []
spec_SVM_fin = [] 
sens_SVM_fin = []
accuracy_SVM_fin = []
accuracy_SVM_sign = []
perm_importances_dfs = []
sign_features_dfs = []

# Define the necessary figures
_, axis_RF_all = plt.subplots()
_, axis_RF_fin = plt.subplots()
_, axis_SVM_fin = plt.subplots()

# Define data and labels
labels = df_comb['Label']
data = df_comb.drop(['Pt_no', 'Label'], axis=1)

# Define ordinal, binary and continuous keys
ordinal_keys = ['Anti-epileptics', 'Psychofarmaca', 'Antacids', 'Anti-hypertensives', 'VitB12', 'Iron-tablets', 'Specialisms_hospitalization', 'Radiologic_investigations', 'Total_amount_ICD10s']
binary_keys = list(df_spec.keys()) + list(df_phecodes.keys()) + list(df_brieven.keys())
binary_keys.remove('Pt_no')
binary_keys.remove('Pt_no')
binary_keys.remove('Pt_no')
#continuous_keys = ['Length', 'BMI', 'Opnames_spec', 'Beeldvormende_verr', 'HR', 'RRsyst', 'RRdiast', 'Vrij T4', 'Hemolytische index', 'Icterische index', 'Lipemische index', 'TSH', 'Alk.Fosf.', 'ALAT', 'ASAT', 'Calcium', 'CKD-EPI eGFR', 'Glucose/PL', 'Hemoglobine', 'Kalium', 'Kreatinine', 'LDH', 'MCV', 'Natrium', 'RDW', 'Tot. Bilirubine', 'Gamma-GT', 'Ureum']
continuous_keys = ['Length', 'Opnames_spec', 'Beeldvormende_verr', 'HR', 'RRsyst', 'RRdiast', 'Vrij T4', 'Hemolytische index', 'Icterische index', 'Lipemische index', 'TSH', 'Alk.Fosf.', 'ALAT', 'ASAT', 'Calcium', 'CKD-EPI eGFR', 'Glucose/PL', 'Hemoglobine', 'Kalium', 'Kreatinine', 'LDH', 'MCV', 'Natrium', 'RDW', 'Tot. Bilirubine', 'Gamma-GT', 'Ureum']
# Test data splitten uit totale dataset
train_data2, test_data2 = train_test_split(df_comb, test_size=0.1, random_state=25)
train_data = train_data2.drop(['Pt_no', 'Label'], axis=1)
test_data = test_data2.drop(['Pt_no', 'Label'], axis=1)
train_label = train_data2['Label']
test_label = test_data2['Label']

# Define 10-fold stratified cross-validation
cv_10fold = model_selection.StratifiedKFold(n_splits=10)

for i, (train_index, val_index) in enumerate(cv_10fold.split(train_data, train_label)):    # Split the data in a train and validation set in a 10-fold cross-validation
    data_train = train_data.iloc[train_index]
    label_train = train_label.iloc[train_index]
    data_val = train_data.iloc[val_index]
    label_val = train_label.iloc[val_index]
 
    # Pre-processing steps
    # Impute data, but exclude the BMI in this. Add BMI again after imputation.
    impute_train, impute_val = impute_data(data_train.loc[:,data_train.columns!='BMI'], data_val.loc[:,data_val.columns !='BMI'], df_decimal) #imputes data for all columns except BMI
    #print(data_train['BMI'])
    #impute_train, impute_val = impute_data(data_train, data_val, df_decimal) # later naar terugkomen
    sam_train = impute_train.join(data_train['BMI'])
    sam_val = impute_val.join(data_val['BMI'])
    print(sam_train)
    # Find significant features per fold
    #sign, sign_features_dfs = find_sign_features(impute_train, label_train, train_index, ordinal_keys, binary_keys, continuous_keys, sign_features_dfs)
    sign, sign_features_dfs = find_sign_features(sam_train, label_train, train_index, ordinal_keys, binary_keys, continuous_keys, sign_features_dfs)

    # Make new dataframes with the significant features
    # train_sign=impute_train[sign]
    # val_sign=impute_val[sign]

    # Scale the data
    # scale_train, scale_val = scale_data(impute_train, impute_val, continuous_keys, ordinal_keys)

    # Define classifiers
    clf_RF_all = RandomForestClassifier()

    # Implement wrapper feature selection
    # sfs1 = sfs(clf_RF_all, k_features=5, forward= False, verbose = 1, scoring = 'neg_mean_squared_error')
    # sfs2 = sfs1.fit(impute_train, label_train) # executing feature selection
    # feat_names = list(sfs2.k_feature_names_)
    # print(f'These are the chosen features for this fold: {feat_names}')
    # Create and test three different models: random forest with all features, random forest with significant features only and support vector machine with only significant features
    # Random forest with all features: create model
    #tprs_RF_all, aucs_RF_all, auc_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, gini_RF_all = pipeline_model(impute_train, label_train, impute_val, label_val, clf_RF_all, tprs_RF_all, aucs_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, axis_RF_all)
    tprs_RF_all, aucs_RF_all, auc_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, gini_RF_all = pipeline_model(sam_train, label_train, sam_val, label_val, clf_RF_all, tprs_RF_all, aucs_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, axis_RF_all)

    # Random forest with significant features only: create model
    #tprs_RF_all, aucs_RF_all, auc_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, gini_RF_all = pipeline_model(impute_train[sign], label_train, impute_val[sign], label_val, clf_RF_all, tprs_RF_all, aucs_RF_all, spec_RF_all, sens_RF_all, accuracy_RF_all, axis_RF_all) # met alleen maar significante features (lagere scores)
    # Random forest with all features: Calculate permutation feature importance
    #result = permutation_importance(clf_RF_all, impute_val[sign], label_val, n_repeats=10, random_state=42, n_jobs=2) # met alleen maar significante features (lagere scores)
    result = permutation_importance(clf_RF_all, impute_val, label_val, n_repeats=10, random_state=42, n_jobs=2) 

    # Create dataframe to store the results
    #df_feature_importance = pd.DataFrame({'Feature': (list(impute_train[sign].columns)), 'Feature importance mean': result.importances_mean, 'Feature importance std': result.importances_std})
    df_feature_importance = pd.DataFrame({'Feature': (list(sam_train.columns)), 'Feature importance mean': result.importances_mean, 'Feature importance std': result.importances_std}) # met alleen maar significante features (lagere scores)

    # Sort dataframe with the most important features first. Keep only the 5 most important features with .head()
    df_feature_importance_sorted = df_feature_importance.sort_values(by=['Feature importance mean'], ascending=False).head()
    # Append dataframe to list per fold. The list consists of i dataframes for the number of folds, showing the best 5 features per fold. This dataframe can be used for visualization.
    perm_importances_dfs.append(df_feature_importance_sorted)

    print(f'This is fold {i}')

# Now, create a dataframe with all duplicate features removed
rel_features_df = pd.DataFrame()
for fold in perm_importances_dfs:
    rel_features_df = pd.concat([rel_features_df, fold])
rel_features_df = rel_features_df.drop_duplicates(subset=['Feature'])
# Make a list of the relevant features
rel_features = rel_features_df['Feature'].tolist()
print(f'These are the relevant features {rel_features}')
# Next, create new dataframes of the training and test data with only these relevant features and preprocess the data)
# Pre-processing steps
# Impute data
impute_train_rel, impute_test_rel = impute_data(train_data, test_data, df_decimal)
# Select only the relevant features
impute_train_fin = impute_train_rel[rel_features]
impute_test_fin = impute_test_rel[rel_features]

# Scale the data
scale_train_rel, scale_test_rel = scale_data(impute_train_rel, impute_test_rel, continuous_keys, ordinal_keys)
# Again, only select the relevant features
scale_train_fin = scale_train_rel[rel_features]
scale_test_fin = scale_test_rel[rel_features]

# Train a SVM and RF classifier using all the training data and validate on the remaining 10% of unseen data
# Define classifiers
clf_RF_fin = RandomForestClassifier()
clf_SVM_fin = SVC()

# Random forest with significant features only: create model
tprs_RF_fin, aucs_RF_fin, auc_RF_fin, spec_RF_fin, sens_RF_fin, accuracy_RF_fin, gini_RF_fin = pipeline_model(impute_train_fin, train_label, impute_test_fin, test_label, clf_RF_fin, tprs_RF_fin, aucs_RF_fin, spec_RF_fin, sens_RF_fin, accuracy_RF_fin, axis_RF_fin)

# Support vector machine with significant features only: create model with scaled data
tprs_SVM_fin, aucs_SVM_fin, auc_SVM_fin, spec_SVM_fin, sens_SVM_fin, accuracy_SVM_fin, gini_SVM_fin = pipeline_model(scale_train_fin, train_label, scale_test_fin, test_label, clf_SVM_fin, tprs_SVM_fin, aucs_SVM_fin, spec_SVM_fin, sens_SVM_fin, accuracy_SVM_fin, axis_SVM_fin)

# Combine true positive rates, areas under curve and axes for plotting mean ROC curves
# all_tprs = [tprs_RF_all, tprs_RF_fin, tprs_SVM_fin]
# all_aucs = [aucs_RF_all, aucs_RF_fin, aucs_SVM_fin]
# all_axes = [axis_RF_all, axis_SVM_fin, axis_models]

# Create plots of the ROC curves for the three models seperately and the mean ROC curves of the three models in one figure
#mean_ROC_curves(all_tprs, all_aucs, all_axes)
#get_ipython().run_line_magic('matplotlib', 'inline')

plt.show()

# Create a dictionary of the scores for the two models. Create dataframe for visualisation.
dict_scores = {'Model 1: Random Forest':[f'{np.round(accuracy_RF_fin, decimals = 2)}',
                                        f'{np.round(sens_RF_fin, decimals=2)}',
                                        f'{np.round(spec_RF_fin, decimals=2)}',
                                        f'{np.round(aucs_RF_fin,decimals=2)}',
                                        f'{np.round(gini_RF_fin,decimals=2)}'],
            'Model 2: Support Vector Machine':[f'{np.round(accuracy_SVM_fin,decimals=2)}',
                                        f'{np.round(sens_SVM_fin,decimals=2)}',
                                        f'{np.round(spec_SVM_fin,decimals=2)}',
                                        f'{np.round(aucs_SVM_fin,decimals=2)}',
                                        f'{np.round(gini_SVM_fin,decimals=2)}']}

df_scores = pd.DataFrame.from_dict(dict_scores, orient='index', columns=['Accuracy', 'Sensitivity', 'Specificity', 'Area under ROC-curve','Gini index'])

print(df_scores)

