import argparse
import os
import math
import numpy as np
import pandas as pd
import skopt
from datetime import datetime
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, train_test_split
from sklearn.linear_model import SGDClassifier
from refdnn.dataset import DATASET, read_cancertypeFile


def get_args():
    parser = argparse.ArgumentParser()
    ## positional
    parser.add_argument('responseFile', type=str, help="A filepath of drug response data for TRAINING")
    parser.add_argument('expressionFile', type=str, help="A filepath of gene expression data for TRAINING")
    parser.add_argument('fingerprintFile', type=str, help="A filepath of fingerprint data for TRAINING")
    parser.add_argument('annotationFile', type=str, help="A filepath of cellline annotation data for TRAINING")
    ## optional
    parser.add_argument('-o', metavar='outputdir', type=str, default='output_2', help="A directory path for saving outputs (default:'output_2')")
    parser.add_argument('-v', metavar='verbose', type=int, default=1, help="0:No logging, 1:Basic logging to check process, 2:Full logging for debugging (default:1)")
    return parser.parse_args()

def main():
    args = get_args()
    outputdir = args.o
    verbose = args.v

    if verbose > 0:
        print('[START]')

    if verbose > 1:
        print('[ARGUMENT] RESPONSEFILE: {}'.format(args.responseFile))
        print('[ARGUMENT] EXPRESSIONFILE: {}'.format(args.expressionFile))
        print('[ARGUMENT] FINGERPRINTFILE: {}'.format(args.fingerprintFile))
        print('[ARGUMENT] OUTPUTDIR: {}'.format(args.o))
        print('[ARGUMENT] VERBOSE: {}'.format(args.v))

    ## output directory
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

    ########################################################
    ## 1. Read data
    ########################################################
    responseFile = args.responseFile
    expressionFile = args.expressionFile
    fingerprintFile = args.fingerprintFile
    annotationFile = args.annotationFile

    dataset = DATASET(responseFile, expressionFile, fingerprintFile)
    if verbose > 0:
        print('[DATA] NUM_PAIRS: {}'.format(len(dataset)))
        print('[DATA] NUM_DRUGS: {}'.format(len(dataset.get_drugs(unique=True))))
        print('[DATA] NUM_CELLS: {}'.format(len(dataset.get_cells(unique=True))))
        print('[DATA] NUM_GENES: {}'.format(len(dataset.get_genes())))
        print('[DATA] NUM_SENSITIVITY: {}'.format(np.count_nonzero(dataset.get_labels()==0)))
        print('[DATA] NUM_RESISTANCE: {}'.format(np.count_nonzero(dataset.get_labels()==1)))

    cell2cancer = read_cancertypeFile(annotationFile)
    cancertypes = [cell2cancer.get(cell, 'others') for cell in dataset.get_cells()]

    ## time log
    timeformat = '[TIME] [{0}] {1.year}-{1.month}-{1.day} {1.hour}:{1.minute}:{1.second}'
    if verbose > 0:
        print(timeformat.format(1, datetime.now()))


    #######################################################
    ## 2. Train Elastic Net using the best hyperparameters
    ########################################################


    ## 2-1) init lists for metrics
    ACCURACY_outer = []
    AUCROC_outer = []
    AUCPR_outer = []
    Precision_outer = []
    Recall_outer = []
    F1_outer = []
    CANCER_outer = []
    Train_size = []
    Test_size = []

    kf = LeaveOneGroupOut()
    n_splits = kf.get_n_splits(groups=cancertypes)
    print("LeaveOneGroupOut.get_n_splits: {}".format(n_splits))
    for k, (idx_train, idx_test) in enumerate(kf.split(X=np.zeros(len(dataset)), groups=cancertypes)):
        ## 2-2) Check a cancer type in test
        test_cancertype = np.unique([cancertypes[x] for x in idx_test])[0]
        CANCER_outer.append(test_cancertype)
        print('[{}/{}] TEST_CANCER: {}'.format(k+1, n_splits, test_cancertype))
        train_num, test_num = len(idx_train), len(idx_test)
        Train_size.append(train_num)
        Test_size.append(test_num)
        print('[{}/{}] training in {} samples and testing in {} samples'.format(k+1, n_splits, train_num, test_num))

        ## 2-3) Set the best values of hyperparameters
        BEST_ALPHA = 0.1
        BEST_L1_RATIO = 0.6079523229995318
        BEST_ETA0 = 0.00039081152125959215

        ## 2-4) Dataset
        idx_train_train, idx_train_valid = train_test_split(idx_train, test_size=0.2, stratify=dataset.get_drugs()[idx_train])
        base_drugs = np.unique(dataset.get_drugs()[idx_train_train])

        X_train = dataset.make_xdata(idx_train_train)
        Y_train = dataset.make_ydata(idx_train_train).ravel()

        X_valid = dataset.make_xdata(idx_train_valid)
        Y_valid = dataset.make_ydata(idx_train_valid).ravel()

        X_test = dataset.make_xdata(idx_test)
        Y_test = dataset.make_ydata(idx_test).ravel()

        ## 2-5) Create a model using the best parameters
        if verbose > 0:
            print('[{}/{}] NOW TRAINING THE MODEL WITH BEST PARAMETERS...'.format(k+1, n_splits))

        clf = SGDClassifier(loss='log',
                            penalty='elasticnet',
                            tol=1e-3,
                            max_iter=1000,
                            early_stopping=True,
                            validation_fraction=0.2,
                            n_jobs=8,
                            alpha=BEST_ALPHA,
                            l1_ratio=BEST_L1_RATIO,
                            eta0=BEST_ETA0)

        ## 2-6) Fit a model
        history = clf.fit(X_train, Y_train)

        ## 2-7) Compute the metric
        Pred_test = clf.predict(X_test)
        Prob_test = clf.predict_proba(X_test)[:,1]

        ACCURACY_outer_k = accuracy_score(Y_test, Pred_test)
        ACCURACY_outer.append(ACCURACY_outer_k)

        AUCROC_outer_k = roc_auc_score(Y_test, Prob_test)
        AUCROC_outer.append(AUCROC_outer_k)

        Precision_outer_k = precision_score(Y_test, Pred_test)
        Precision_outer.append(Precision_outer_k)

        Recall_outer_k = recall_score(Y_test, Pred_test)
        Recall_outer.append(Recall_outer_k)

        F1_outer_k = f1_score(Y_test, Pred_test)
        F1_outer.append(F1_outer_k)

        AUCPR_outer_k = average_precision_score(Y_test, Prob_test)
        AUCPR_outer.append(AUCPR_outer_k)

        if verbose > 0:
            print('[{}/{}] TEST_ACCURACY : {:.3f}'.format(k+1, n_splits, ACCURACY_outer_k))
            print('[{}/{}] TEST_AUCROC : {:.3f}'.format(k+1, n_splits, AUCROC_outer_k))
            print('[{}/{}] TEST_Precision : {:.3f}'.format(k+1, n_splits, Precision_outer_k))
            print('[{}/{}] TEST_Recall : {:.3f}'.format(k+1, n_splits, Recall_outer_k))
            print('[{}/{}] TEST_F1 : {:.3f}'.format(k+1, n_splits, F1_outer_k))
            print('[{}/{}] TEST_AUCPR : {:.3f}'.format(k+1, n_splits, AUCPR_outer_k))

        ## time log
        if verbose > 0:
            print(timeformat.format(3, datetime.now()))

    #######################################################
    ## 3. Save the results
    ########################################################
    res = pd.DataFrame.from_dict({'CANCERTYPE':CANCER_outer,
                                  'Train_size':Train_size,
                                  'Test_size':Test_size,
                                  'Accuracy':ACCURACY_outer,
                                  'Precision':Precision_outer,
                                  'Recall':Recall_outer,
                                  'F1':F1_outer,
                                  'AUCROC':AUCROC_outer,
                                  'AUCPR':AUCPR_outer,
                                  'ALPHA':BEST_ALPHA,
                                  'L1_RATIO':BEST_L1_RATIO,
                                  'ETA0':BEST_ETA0})
    #res = res[['CANCERTYPE', 'Train_size', 'Test_size', 'ACCURACY', 'AUCROC', 'AUCPR', 'ALPHA', 'L1_RATIO', 'ETA0']]
    res.to_csv(os.path.join(outputdir, 'metrics_hyperparameters.csv'), sep=',')

    ## time log
    if verbose > 0:
        print(timeformat.format(4, datetime.now()))

    if verbose > 0:
        print('[FINISH]')


if __name__=="__main__":
    main()
