import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
from scipy import interp
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler
import operator
from sklearn.metrics import balanced_accuracy_score

#AUC ROC Curve Scoring Function for Multi-class Classification
#"macro"
#"weighted"
# None
def multiclass_roc_auc_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return roc_auc_score(y_test, y_probs, average=average)

def multiclass_average_precision_score(y_test, y_probs, average="macro"):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    return average_precision_score(y_test, y_probs, average=average)

def multiclass_roc_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    fpr = dict()
    tpr = dict()
    for i in range(y_probs.shape[1]):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_probs[:, i])
        
    return (fpr, tpr)

def multiclass_average_precision_curve(y_test, y_probs):
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    precision = dict()
    recall = dict()
    for i in range(y_probs.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_probs[:, i])
    
    return (precision, recall)

def Accuracykfold(X,y,splits):
    Xs = np.copy(X)
    ys=np.copy(y)
    numfolds=5;
    numlabels=4;
    performancesAccuracy=np.empty([numfolds, 1]);
    
    
    index=0
    for train, test in splits:
        # print("%s %s" % (train, test))
        clf = RandomForestClassifier(n_estimators = 200, max_features='sqrt', max_depth=420,random_state=0)
        #{'n_estimators': 200, 'max_features': 'sqrt', 'max_depth': 420}
        clf.fit(Xs[train,:], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test,:])
        performancesAccuracy[index]=accuracy_score(ys[test], y_pred)
        index+=1

    return performancesAccuracy

# returns performances and splits/models used in the cross-validation
from sklearn.metrics import balanced_accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def AUCAUPkfold(X, y, smoteflag, verbose=True):
    numfolds = 5
    numlabels = 4
    cv = StratifiedKFold(n_splits=numfolds, shuffle=True)
    Xs = np.copy(X)
    ys = np.copy(y)

    if smoteflag:
        smote = SMOTE()
        Xs, ys = smote.fit_resample(Xs, ys)

    performancesAUC = np.empty([numfolds, numlabels])
    performancesAUP = np.empty([numfolds, numlabels])
    performancesBalancedAcc = np.empty(numfolds)
    splits = []
    model_per_fold = []
    index = 0

    for train, test in cv.split(Xs, ys):
        clf = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=420, random_state=0)

        splits.append([train, test])
        clf.fit(Xs[train, :], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test, :])
        y_probs = clf.predict_proba(Xs[test, :])

        performancesAUC[index, :] = np.array(multiclass_roc_auc_score(ys[test], y_probs, average=None))
        performancesAUP[index, :] = np.array(multiclass_average_precision_score(ys[test], y_probs, average=None))
        performancesBalancedAcc[index] = balanced_accuracy_score(ys[test], y_pred)

        index += 1
        model_per_fold.append(clf)

    if verbose:
        print("AUC: average over the folds")
        print(performancesAUC.mean(axis=0))
        print("AUC: std over the folds")
        print(performancesAUC.std(axis=0))

        print("AUP: average over the folds")
        print(performancesAUP.mean(axis=0))
        print("AUP: std over the folds")
        print(performancesAUP.std(axis=0))

        print("Balanced Accuracy: average over the folds")
        print(performancesBalancedAcc.mean())
        print("Balanced Accuracy: std over the folds")
        print(performancesBalancedAcc.std())

    return (performancesAUC, performancesAUP, splits, model_per_fold)


# Helper functions
def multiclass_roc_auc_score(y_true, y_score, average="macro"):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_score, average=average, multi_class='ovo')


def multiclass_average_precision_score(y_true, y_score, average="macro"):
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_score, average=average)


from sklearn.metrics import balanced_accuracy_score
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def AUCAUPkfold(X, y, smoteflag, verbose=True):
    numfolds = 5
    numlabels = 4
    cv = StratifiedKFold(n_splits=numfolds, shuffle=True)
    Xs = np.copy(X)
    ys = np.copy(y)

    if smoteflag:
        smote = SMOTE()
        Xs, ys = smote.fit_resample(Xs, ys)

    performancesAUC = np.empty([numfolds, numlabels])
    performancesAUP = np.empty([numfolds, numlabels])
    performancesBalancedAcc = np.empty(numfolds)
    splits = []
    model_per_fold = []
    index = 0

    for train, test in cv.split(Xs, ys):
        clf = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=420, random_state=0)

        splits.append([train, test])
        clf.fit(Xs[train, :], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test, :])
        y_probs = clf.predict_proba(Xs[test, :])

        performancesAUC[index, :] = np.array(multiclass_roc_auc_score(ys[test], y_probs, average=None))
        performancesAUP[index, :] = np.array(multiclass_average_precision_score(ys[test], y_probs, average=None))
        performancesBalancedAcc[index] = balanced_accuracy_score(ys[test], y_pred)

        index += 1
        model_per_fold.append(clf)

    if verbose:
        print("AUC: average over the folds")
        print(performancesAUC.mean(axis=0))
        print("AUC: std over the folds")
        print(performancesAUC.std(axis=0))

        print("AUP: average over the folds")
        print(performancesAUP.mean(axis=0))
        print("AUP: std over the folds")
        print(performancesAUP.std(axis=0))

        print("Balanced Accuracy: average over the folds")
        print(performancesBalancedAcc.mean())
        print("Balanced Accuracy: std over the folds")
        print(performancesBalancedAcc.std())

    return (performancesAUC, performancesAUP, splits, model_per_fold)

def AUCAUPkfold_splits(X, y, smoteflag):
    numfolds = 5
    cv = StratifiedKFold(n_splits=numfolds, shuffle=True)
    Xs = np.copy(X)
    ys = np.copy(y)

    if smoteflag:
        smote = SMOTE()
        Xs, ys = smote.fit_resample(Xs, ys)

    splits = []

    for train, test in cv.split(Xs, ys):

        splits.append([train, test])
    return splits

from sklearn.model_selection import StratifiedGroupKFold
def AUCAUPkfold_splits_group(X, y, smoteflag, group_column):
    numfolds = 5
    Xs = X.copy()
    ys = y.copy()

    # Extract groups from the specified group column
    groups = Xs[group_column]

    # Optionally apply SMOTE before splitting
    if smoteflag:
        smote = SMOTE()
        Xs_smote, ys_smote = smote.fit_resample(Xs, ys)
        groups_smote = groups.iloc[smote.sample_indices_]
        Xs = Xs_smote
        ys = ys_smote
        groups = groups_smote.reset_index(drop=True)
    else:
        groups = groups.reset_index(drop=True)

    # Drop the group column from features
    Xs = Xs.drop(columns=[group_column]).reset_index(drop=True)
    ys = ys.reset_index(drop=True)

    splits = []

    # Initialize StratifiedGroupKFold
    cv = StratifiedGroupKFold(n_splits=numfolds, shuffle=True, random_state=0)

    for train_idx, test_idx in cv.split(Xs, ys, groups=groups):
        splits.append([train_idx, test_idx])

    return splits


import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE


# Make sure to define or import these functions as they are used in the code
# from your_module import multiclass_roc_auc_score, multiclass_average_precision_score

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import SMOTE


# Ensure these custom functions are defined or imported appropriately
# from your_module import multiclass_roc_auc_score, multiclass_average_precision_score

def AUCAUPkfold_full(X, y, smote_mode, verbose=True):
    print('full3')
    numfolds = 5
    numlabels = 4  # Adjust if your number of classes is different
    cv = StratifiedKFold(n_splits=numfolds, shuffle=True)
    Xs = X.copy()
    ys = y.copy()

    # Apply SMOTE before splitting if specified
    if smote_mode == 'before_split':
        smote = random_over_sampler()
        Xs, ys = smote.fit_resample(Xs, ys)

    performancesAUC = np.empty([numfolds, numlabels])
    performancesAUP = np.empty([numfolds, numlabels])
    performancesBalancedAcc = np.empty(numfolds)
    splits = []
    model_per_fold = []
    feature_importances = []
    index = 0

    for train_index, test_index in cv.split(Xs, ys):
        X_train, y_train = Xs.iloc[train_index], ys.iloc[train_index]
        X_test, y_test = Xs.iloc[test_index], ys.iloc[test_index]

        # Apply SMOTE to training data after splitting if specified
        if smote_mode in ['train', 'train_test']:
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)

        # Apply SMOTE to test data after splitting if specified
        if smote_mode == 'train_test':
            smote = SMOTE()
            X_test, y_test = smote.fit_resample(X_test, y_test)

        clf = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=420, random_state=0)

        splits.append([train_index, test_index])
        clf.fit(X_train, y_train)
        model_per_fold.append(clf)
        feature_importances.append(clf.feature_importances_)

        # Predicting the Test set results
        y_pred = clf.predict(X_test)
        y_probs = clf.predict_proba(X_test)

        # Ensure that your custom metric functions are defined
        performancesAUC[index, :] = np.array(multiclass_roc_auc_score(y_test, y_probs, average=None))
        performancesAUP[index, :] = np.array(multiclass_average_precision_score(y_test, y_probs, average=None))
        performancesBalancedAcc[index] = balanced_accuracy_score(y_test, y_pred)

        index += 1

    # Compute average feature importances over folds
    avg_feature_importances = np.mean(feature_importances, axis=0)

    if verbose:
        print("\nAUC: average over the folds")
        print(performancesAUC.mean(axis=0))
        print("AUC: std over the folds")
        print(performancesAUC.std(axis=0))

        print("\nAUP: average over the folds")
        print(performancesAUP.mean(axis=0))
        print("AUP: std over the folds")
        print(performancesAUP.std(axis=0))

        print("\nBalanced Accuracy: average over the folds")
        print(performancesBalancedAcc.mean())
        print("Balanced Accuracy: std over the folds")
        print(performancesBalancedAcc.std())

        # Now, print the top important features
        top_n = 10  # Adjust the number of top features you want to display
        indices = np.argsort(avg_feature_importances)[::-1][:top_n]
        print(f"\nTop {top_n} important features:")
        feature_names = X.columns
        for idx in indices:
            print(f"{feature_names[idx]}: {avg_feature_importances[idx]:.4f}")

    return (performancesAUC, performancesAUP, splits, model_per_fold)



# Now run the function four times with different SMOTE settings


# Helper functions
def multiclass_roc_auc_score(y_true, y_score, average=None):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import roc_auc_score
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    return roc_auc_score(y_true_bin, y_score, average=average, multi_class='ovr')


def multiclass_average_precision_score(y_true, y_score, average=None):
    from sklearn.preprocessing import label_binarize
    from sklearn.metrics import average_precision_score
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    return average_precision_score(y_true_bin, y_score, average=average)


from sklearn.metrics import balanced_accuracy_score
from imblearn.over_sampling import RandomOverSampler

from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import label_binarize
import numpy as np


def multiclass_roc_auc_score(y_true, y_score, average="macro"):
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    return roc_auc_score(y_true_bin, y_score, average=average, multi_class='ovr')


from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import RandomOverSampler
import numpy as np


def ROCkfold(X, y, splits, verbose=True):
    Xs = np.copy(X)
    ys = np.copy(y)
    numfolds = len(splits)
    numlabels = len(np.unique(y))

    mean_fpr = np.linspace(0, 1, 500)
    performancesAUC = np.empty([numfolds, numlabels])
    balanced_performancesAUC = np.empty([numfolds, numlabels])  # For Balanced AUC
    performancesROC = [np.empty([numfolds, len(mean_fpr)]) for _ in range(numlabels)]

    index = 0
    for train, test in splits:
        clf = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=420, random_state=0)
        clf.fit(Xs[train], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test])
        y_probs = clf.predict_proba(Xs[test])
        balanced_acc = balanced_accuracy_score(ys[test], y_pred)  # Calculate Balanced Accuracy

        for c in range(numlabels):
            # Binary labels for the current class
            binary_y_test = (ys[test] == c).astype(int)
            # Probabilities for the current class
            y_probs_c = y_probs[:, c]
            # Unbalanced AUC
            auc = roc_auc_score(binary_y_test, y_probs_c)
            performancesAUC[index, c] = auc

            # ROC curve for the current class
            fpr, tpr, _ = roc_curve(binary_y_test, y_probs_c)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            performancesROC[c][index, :] = interp_tpr

            # Apply RandomOverSampler to the test data for the current class
            ros = RandomOverSampler()
            ros_X_test, ros_y_test = ros.fit_resample(Xs[test], binary_y_test)

            # Predict probabilities on the oversampled test set
            ros_y_probs = clf.predict_proba(ros_X_test)
            ros_y_probs_c = ros_y_probs[:, c]

            # Balanced AUC for the current class
            balanced_auc = roc_auc_score(ros_y_test, ros_y_probs_c)
            balanced_performancesAUC[index, c] = balanced_auc

        index += 1

    # Calculate mean and std of tpr for plotting
    mean_tpr = []
    std_tpr = []
    tprs_upper = []
    tprs_lower = []
    for c in range(numlabels):
        mean_tpr_c = np.mean(performancesROC[c], axis=0)
        mean_tpr_c[-1] = 1.0
        std_tpr_c = np.std(performancesROC[c], axis=0)
        tprs_upper_c = np.minimum(mean_tpr_c + std_tpr_c, 1)
        tprs_lower_c = np.maximum(mean_tpr_c - std_tpr_c, 0)
        mean_tpr.append(mean_tpr_c)
        std_tpr.append(std_tpr_c)
        tprs_upper.append(tprs_upper_c)
        tprs_lower.append(tprs_lower_c)

    if verbose:
        print("AUC: average over the folds")
        print(performancesAUC.mean(axis=0))
        print("AUC: std over the folds")
        print(performancesAUC.std(axis=0))

        print("Balanced Accuracy over the folds:")
        print(balanced_acc)

        print("Balanced AUC: average over the folds")
        print(balanced_performancesAUC.mean(axis=0))
        print("Balanced AUC: std over the folds")
        print(balanced_performancesAUC.std(axis=0))

    return performancesAUC, performancesROC, mean_fpr, mean_tpr, std_tpr, tprs_upper, tprs_lower


def ROCplot(mean_fpr, mean_tpr, std_tpr, tprs_upper, tprs_lower, performancesAUC, labelc):
    mean_auc=performancesAUC.mean(axis=0)
    std_auc=performancesAUC.std(axis=0)
                                
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    ax.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="ROC "+"Class "+ str(labelc+1))
    ax.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


# labeld--> label for all the dbs

def ROCMultiPlot(mean_fprL, mean_tprL, std_tprL, tprs_upperL, tprs_lowerL, performancesAUCL, labelc, labeld, colord):
    mean_auc=[p.mean(axis=0) for p in  performancesAUCL]
    std_auc=[p.std(axis=0) for p in  performancesAUCL]

    fig, ax = plt.subplots(figsize=(5, 5))
    #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    for d in range(len(labeld)):
        ax.plot(mean_fprL[d], mean_tprL[d], color=colord[d],label=labeld[d]+ r': AUC = %0.4f $\pm$ %0.4f' % (mean_auc[d], std_auc[d]), lw=2, alpha=.5)
        ax.fill_between(mean_fprL[d], tprs_lowerL[d], tprs_upperL[d], color=colord[d], alpha=.1)

    ax.set(xlim=[0, 1], ylim=[0, 1], title="ROC "+"Class "+ str(labelc+1))        
    ax.legend(loc="lower right")
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()

def ROCMultiPlotCallable(mean_fprL, mean_tprL, std_tprL, tprs_upperL, tprs_lowerL, performancesAUCL, labelc, labeld, colord,ax):
    mean_auc=[p.mean(axis=0) for p in  performancesAUCL]
    std_auc=[p.std(axis=0) for p in  performancesAUCL]
    #ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
    
    for d in range(len(labeld)):
        ax.plot(mean_fprL[d], mean_tprL[d], color=colord[d],label=labeld[d]+ r': AUC = %0.4f $\pm$ %0.4f' % (mean_auc[d], std_auc[d]), lw=2, alpha=.5)
        ax.fill_between(mean_fprL[d], tprs_lowerL[d], tprs_upperL[d], color=colord[d], alpha=.1)
     
    ax.legend(loc="lower right")


from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import label_binarize
import numpy as np


def multiclass_average_precision_curve(y_true, y_score):
    # Binarize the output
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    precision, recall = {}, {}
    for i in range(y_true_bin.shape[1]):
        precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_score[:, i])
    return precision, recall


def multiclass_average_precision_score(y_true, y_score, average="macro"):
    y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
    return average_precision_score(y_true_bin, y_score, average=average)


from sklearn.metrics import precision_recall_curve, average_precision_score
from imblearn.over_sampling import RandomOverSampler
import numpy as np

def PrecisionRecallCurvekfold(X, y, splits, verbose=True):
    Xs = np.copy(X)
    ys = np.copy(y)
    numfolds = len(splits)
    numlabels = len(np.unique(y))

    mean_recall = np.linspace(0, 1, 500)
    performancesAUP = np.empty([numfolds, numlabels])
    balanced_performancesAUP = np.empty([numfolds, numlabels])  # For Balanced AUP
    performancesPrecisionRecall = [np.empty([numfolds, len(mean_recall)]) for _ in range(numlabels)]

    index = 0
    for train, test in splits:
        clf = RandomForestClassifier(n_estimators=200, max_features='sqrt', max_depth=420, random_state=0)
        clf.fit(Xs[train], ys[train])

        # Predicting the Test set results
        y_pred = clf.predict(Xs[test])
        y_probs = clf.predict_proba(Xs[test])
        balanced_acc = balanced_accuracy_score(ys[test], y_pred)  # Calculate Balanced Accuracy

        for c in range(numlabels):
            # Binary labels for the current class
            binary_y_test = (ys[test] == c).astype(int)
            # Probabilities for the current class
            y_probs_c = y_probs[:, c]
            # Unbalanced AUP
            aup = average_precision_score(binary_y_test, y_probs_c)
            performancesAUP[index, c] = aup

            # Precision-Recall curve for the current class
            precision, recall, _ = precision_recall_curve(binary_y_test, y_probs_c)
            interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
            performancesPrecisionRecall[c][index, :] = interp_precision

            # Apply RandomOverSampler to the test data for the current class
            ros = RandomOverSampler()
            ros_X_test, ros_y_test = ros.fit_resample(Xs[test], binary_y_test)

            # Predict probabilities on the oversampled test set
            ros_y_probs = clf.predict_proba(ros_X_test)
            ros_y_probs_c = ros_y_probs[:, c]

            # Balanced AUP for the current class
            balanced_aup = average_precision_score(ros_y_test, ros_y_probs_c)
            balanced_performancesAUP[index, c] = balanced_aup

        index += 1

    # Calculate mean and std of precision for plotting
    mean_precision = []
    std_precision = []
    precision_upper = []
    precision_lower = []
    for c in range(numlabels):
        mean_precision_c = np.mean(performancesPrecisionRecall[c], axis=0)
        std_precision_c = np.std(performancesPrecisionRecall[c], axis=0)
        precision_upper_c = np.minimum(mean_precision_c + std_precision_c, 1)
        precision_lower_c = np.maximum(mean_precision_c - std_precision_c, 0)
        mean_precision.append(mean_precision_c)
        std_precision.append(std_precision_c)
        precision_upper.append(precision_upper_c)
        precision_lower.append(precision_lower_c)

    if verbose:
        print("Balanced AUP: average over the folds")
        print(balanced_performancesAUP.mean(axis=0))
        print("Balanced AUP: std over the folds")
        print(balanced_performancesAUP.std(axis=0))

        print("Balanced Accuracy over the folds:")
        print(balanced_acc)

        print("AUP: average over the folds")
        print(performancesAUP.mean(axis=0))
        print("AUP: std over the folds")
        print(performancesAUP.std(axis=0))

    return performancesAUP, performancesPrecisionRecall, mean_recall, mean_precision, std_precision, precision_upper, precision_lower


def PrecisionRecallplot(mean_recall, mean_precision, std_precision, precision_upper, precision_lower, performancesAUP, labelc):
    mean_aup=performancesAUP.mean(axis=0)
    std_aup=performancesAUP.std(axis=0)
                                
    fig, ax = plt.subplots()
    ax.plot(mean_recall, mean_precision, color='b',label=r'Mean Precision Recall (AUP = %0.2f $\pm$ %0.2f)' % (mean_aup, std_aup), lw=2, alpha=.8)
    ax.fill_between(mean_recall, precision_lower, precision_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Precision-Recall Curve "+ "Class "+ str(labelc +1))
    ax.legend(loc="lower left")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


# labeld--> label for all the dbs
def PrecisionRecallMultiPlot(mean_recallL, mean_precisionL, std_precisionL, precision_upperL, precision_lowerL, performancesAUPL, labelc, labeld, colord):
    mean_aup=[p.mean(axis=0) for p in  performancesAUPL]
    std_aup=[p.std(axis=0) for p in  performancesAUPL]
    
    fig, ax = plt.subplots(figsize=(5, 5))
    
    for d in range(len(labeld)):
        ax.plot(mean_recallL[d], mean_precisionL[d], color=colord[d],label=labeld[d]+ r': AUP = %0.4f $\pm$ %0.4f' % (mean_aup[d], std_aup[d]), lw=2, alpha=.5)
        ax.fill_between(mean_recallL[d], precision_lowerL[d], precision_upperL[d], color=colord[d], alpha=.1)



    ax.set(xlim=[0, 1], ylim=[0, 1], title="Precision-Recall Curve "+ "Class "+ str(labelc +1)) 
    ax.legend(loc="lower left")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.show()


def PrecisionRecallMultiPlotCallable(mean_recallL, mean_precisionL, std_precisionL, precision_upperL, precision_lowerL, performancesAUPL, labelc, labeld, colord,ax):
    mean_aup=[p.mean(axis=0) for p in  performancesAUPL]
    std_aup=[p.std(axis=0) for p in  performancesAUPL]
    
    for d in range(len(labeld)):
        ax.plot(mean_recallL[d], mean_precisionL[d], color=colord[d],label=labeld[d]+ r': AUP = %0.4f $\pm$ %0.4f' % (mean_aup[d], std_aup[d]), lw=2, alpha=.5)
        ax.fill_between(mean_recallL[d], precision_lowerL[d], precision_upperL[d], color=colord[d], alpha=.1)

    ax.legend(loc="lower left")


