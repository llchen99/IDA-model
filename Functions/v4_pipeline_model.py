import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
# functie zonder te plotten

def pipeline_model4(train_data, train_label, test_data, test_label, clf):
    '''In this function, a machine learning model is created and tested. Dataframes of the train data, train labels, test data and test labels
    must be given as input. Also, the classifier must be given as input. Scoring metrics true positives, area under curve, specificity, sensitivity
    and accuracy must be given as input, these scores are appended every fold and are returned. The axis must also be given in order to plot the ROC curves
    for the different folds in the right figure.'''
    # Fit and test the classifier
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)

    return tprs, aucs, auc, spec, sens, accuracy, gini
