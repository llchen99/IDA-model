import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RandomizedSearchCV

# Voor SVM
def pipeline_model2(train_data, train_label, test_data, test_label, clf, parameters, tprs, aucs, spec, sens, accuracy, axis, n_estimators_sel, max_features_sel, max_depth_sel, min_samples_split_sel, min_samples_leaf_sel, bootstrap_sel):
    '''In this function, a machine learning model is created and tested. Dataframes of the train data, train labels, test data and test labels
    must be given as input. Also, the classifier must be given as input. Scoring metrics true positives, area under curve, specificity, sensitivity
    and accuracy must be given as input, these scores are appended every fold and are returned. The axis must also be given in order to plot the ROC curves
    for the different folds in the right figure.'''
    # Fit and test the classifier
    clf = RandomizedSearchCV(clf, parameters)
    clf.fit(train_data, train_label)
    predicted = clf.predict(test_data)

    # Append the chosen hyperparameters
    n_estimators_sel.append(clf.best_estimator_.n_estimators)
    max_features_sel.append(clf.best_estimator_.max_features)
    max_depth_sel.append(clf.best_estimator_.max_depth)
    min_samples_split_sel.append(clf.best_estimator_.min_samples_split)
    min_samples_leaf_sel.append(clf.best_estimator_.min_samples_leaf)
    bootstrap_sel.append(clf.best_estimator_.bootstrap)

    # plot ROC-curve per fold
    mean_fpr = np.linspace(0, 1, 100)    # Help for plotting the false positive rate
    viz = metrics.plot_roc_curve(clf, test_data, test_label, name='ROC fold {}'.format(1), alpha=0.3, color = "navy", lw=1, ax=axis)    # Plot the ROC-curve for this fold on the specified axis.
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)    # Interpolate the true positive rate
    interp_tpr[0] = 0.0    # Set the first value of the interpolated true positive rate to 0.0
    tprs.append(interp_tpr)   # Append the interpolated true positive rate to the list
    aucs.append(viz.roc_auc)    # Append the area under the curve to the list
    auc = viz.roc_auc
    gini = (2*auc)-1


    # Calculate the scoring metrics
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel()   # Find the true negatives, false positives, false negatives and true positives from the confusion matrix
    spec.append(tn/(tn+fp))    # Append the specificity to the list
    sens.append(tp/(tp+fn))    # Append the sensitivity to the list
    accuracy.append(metrics.accuracy_score(test_label, predicted))    # Append the accuracy to the list
    return tprs, aucs, auc, spec, sens, accuracy, gini, n_estimators_sel, max_features_sel, max_depth_sel, min_samples_split_sel, min_samples_leaf_sel, bootstrap_sel
