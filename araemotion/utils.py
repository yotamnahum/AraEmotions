import pandas as pd
import numpy as np
from .preproccess import simple_clean

def MakeMultiLabel(data,labels_list):
    df = data.copy()
    label_array = df[labels_list].astype(int).values
    df['labels'] = pd.Series(object)
    df['labels'] = label_array.tolist()
    output_columns = ['text', 'labels']
    df = df[output_columns].copy()
    return df

def text_cleaner(text):
    if type(text) is str:
        clean_text = simple_clean(text)
    elif (type(text) is list) or (type(text) is pd.Series):
        clean_text = [simple_clean(i) for i  in text]
    return clean_text

def prepare_data(data: pd.DataFrame,labels_list):
    df = pd.DataFrame()
    df["text"] = text_cleaner(data["text"])
    df["labels"] = MakeMultiLabel(data,labels_list)["labels"]
    return df

def get_proba_df(predictions, raw_outputs, emotion_list):
    pred = pd.DataFrame(predictions,columns=emotion_list)
    pred_labels = pred.dot(pd.Index(emotion_list) + ', ' ).str.strip(', ')
    proba_df = pd.DataFrame(raw_outputs,columns=emotion_list)
    proba_df['pred_labels'] = pred_labels
    return proba_df