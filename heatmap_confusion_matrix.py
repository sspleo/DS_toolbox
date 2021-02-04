import matplotlib.pyplot as plt
from pandas import DataFrame as df
from pandas_confusion import BinaryConfusionMatrix
from seaborn import heatmap

def heatmap_confusion_matrix(actual_label, pred_label, output_folder, plot_title):
    """  Plot Confusion Matrix using Seaborn's Heatmap.
    
    This plot contains not just confusion matrix but also
        Accuracy, TPR(Recall, Sensitivity), Precision, TNR(specificity) and F1-Score
    Columns are predicted labels and Rows are actual labels
    
    Parameters
    ----------
    actual_label : 1D array
        actual labels of the data
    pred_label : 1D array
        predicted labels
    output_folder : path
        path to output folder where output plot will be saved
    plot_title : string
        plot title which may conclude data information and model description
    
    Returns
    -------
    result : Confusion matrix plot with test result statistics
            Saved plot file to output_folder
    """
    # Create confusion matrix
    binary_confusion_matrix = BinaryConfusionMatrix(actual_label, pred_label)
    
    # Result statistics from the confusion matrix
    stats = binary_confusion_matrix.stats()
    pos_real = stats['P']
    neg_real = stats['N']
    pos_pred = stats['PositiveTest']
    neg_pred = stats['NegativeTest']
    TP = stats['TP']
    TN = stats['TN']
    FP = stats['FP']
    FN = stats['FN']
    TPR = round(stats['TPR'], 2)    #sensitivity, recall: TP/(TP+FN) = TP/pos_real
    TNR = round(stats['TNR'], 2)    #specificity
    PPV = round(stats['PPV'], 2)    #precision : TP/(TP+FP) = TP/pos_pred
    F1_score = round(stats['F1_score'], 2) #harmonic mean of recall and precision
    ACC = round(stats['ACC'], 2)
    
    # Confusion matrix for display
    cm = np.array([[TN,FP], [FN,TP]])
    """
        TN  FP
        FN  TP
    """
    df_cm = df(cm, index = ['{}  \nDecoy'.format(neg_real), '%d  \nActive'%(pos_real)], columns = ['Decoy\n%d'%(neg_pred), 'Active\n%d'%(pos_pred)])
    plot = plt.figure(figsize = (6, 6))
    plt.rcParams['font.size'] = 10
    plt.title("Accuracy : {:.2f}   TPR : {:.2f}\nPrecision : {:.2f}   TNR : {:.2f}   F1-Score : {:.2f}".format(ACC, TPR, PPV, TNR, F1_score), loc = 'left', fontsize = 12)
    plt.suptitle(plot_title, y = 0.95, fontsize= 14)
    plt.xlabel('Predicted Label')
    plt.ylabel('Actual Label')
    plt.subplots_adjust(top = 0.8)
    
    # plot heatmap
    heatmap(df_cm, annot = True, fmt = 'd', annot_kws={"size":14})
    
    # save plot and display it
    plot.savefig('{}/test_result_confusion_matrix.png'.format(output_folder))
    plt.show()
    plt.close()


"""
Examples
--------
>>>> import numpy as np
>>>> import os
>>>> reals = np.array([1,1,1,1,1,1,1,1,1,1, #10
                0,0,0,0,0,0,0,0,0, #9
                0,0, #2
                1,1,1]) #3
>>>> predicts = np.array([1,1,1,1,1,1,1,1,1,1,
                    0,0,0,0,0,0,0,0,0,
                    1,1,
                    0,0,0])
>>>> heatmap_confusion_matrix(reals, predicts, os.getcwd(), 'Heatmap Confusion Matrix Sample')
"""
