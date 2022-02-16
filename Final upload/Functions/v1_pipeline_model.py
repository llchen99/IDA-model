import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


def pipeline_model(train_data, train_label, test_data, test_label, clf, tprs, aucs, spec, sens, accuracy, axis, gini, j):
    '''In this function, a machine learning model is created and tested.
    Inputs:
        train_data: dataframe of training data
        train_label: dataframe of training labels
        test_data: dataframe of test data
        test_label: dataframe of test labels
        clf: defined classifier
        tprs: list of true positive rates
        aucs: list of areas under the curve
        spec: list of specificity
        sens: list of sensitivity
        accuracy: list of accuracies
        axis: axis
        gini: list of gini scores
        j: fold of outer fold
    Returns:
        tprs: new list of true positive rates
        aucs: new list of areas under the curve
        auc: area under the curve of that fold
        spec: new list of specificities
        sens: new list of sensitivities
        accuracy: new list of accuracies
        gini: new list of gini indices
    '''
    # Fit and test the classifier
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)

    # plot ROC-curve per fold
    mean_fpr = np.linspace(0, 1, 100)    # Help for plotting the false positive rate
    viz = metrics.plot_roc_curve(clf, test_data, test_label, name=f'ROC fold {j+1}', alpha=0.3,
                                 color="navy", lw=1, ax=axis)    # Plot the ROC-curve for this fold on the specified axis.
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)    # Interpolate the true positive rate
    interp_tpr[0] = 0.0    # Set the first value of the interpolated true positive rate to 0.0
    tprs.append(interp_tpr)   # Append the interpolated true positive rate to the list
    aucs.append(viz.roc_auc)    # Append the area under the curve to the list
    auc = viz.roc_auc

    # Calculate the scoring metrics
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()
    spec.append(tn/(tn+fp))  # Append the specificity to the list
    sens.append(tp/(tp+fn))    # Append the sensitivity to the list
    accuracy.append(metrics.accuracy_score(test_label, predicted))  # Append the accuracy to the list
    gini.append((2*auc)-1)  # append gini to the list
    return tprs, aucs, auc, spec, sens, accuracy, gini
