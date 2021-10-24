import pandas as pd
from sklearn.metrics import classification_report,precision_score
import numpy as np

def macro_accuracy(threshold: float = None,
                   emotion_list: list = None):

    def _threshold(x, threshold):
        if x >= threshold:
            return 1
        return 0

    def macro_acc(y_true,y_pred):
        threshold_value = threshold
        pred_labels = [[_threshold(i,threshold_value) for i in pred] for pred in y_pred]
        print(classification_report(y_true,pred_labels,target_names=emotion_list,zero_division=0))
        return precision_score(y_true, pred_labels, average='macro')

    return macro_acc