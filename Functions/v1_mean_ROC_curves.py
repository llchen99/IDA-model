import numpy as np
from sklearn import metrics

def mean_ROC_curves(tprs_all, aucs_all, axis_all):
    '''With this function, the mean ROC-curves of the models over a 10-cross-validation are plot.
    The true positive rates, areas under the curve and axes where the mean ROC-curve must be plot
    are given as input for different models. The figures are filled with the mean and std ROC-curve and
    can be visualized with plt.show()'''
    for i, (tprs, aucs, axis) in enumerate(zip(tprs_all, aucs_all, axis_all[:2])):   # Loop over the tprs, aucs and first three axes for the figures of the three different models.
        # Calculate means and standard deviations of true positive rate, false positive rate and area under curve
        names = ['Random forest', 'Support Vector Machine']
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_fpr = np.linspace(0, 1, 100)
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        std_tpr = np.std(tprs, axis=0)
        axis.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)   # Plot the mean ROC-curve for the corresponding model
        axis_all[2].plot(mean_fpr, mean_tpr, label=fr'Mean ROC model {(names[i])} (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)    # Plot the mean ROC-curve for the corresponding model in another figure
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)    # Set the upper value of the true positive rates
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)    # Set the upper value of the true positive rates
        axis.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')    # Plot the standard deviations of the ROC-curves
        axis.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title=f'ROC-curves model {names[i]}')    # Set axes and title
        axis.legend(loc="lower right")    # Set legend
        axis_all[2].fill_between(mean_fpr, tprs_lower, tprs_upper, alpha=.2, label=r'$\pm$ 1 std. dev.')    # Plot the standard deviations of the ROC-curves in another figure
        axis_all[2].set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title='Mean ROC-curve for the two models')    # Set axes and title
        axis_all[2].legend(loc="lower right")    # Set legend
    return