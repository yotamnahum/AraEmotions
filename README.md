# Emotion Analysis on Arabic Tweets: Research Implementation

This repository contains the implementation of key research findings in the field of emotion analysis on Arabic tweets. The code here is developed based on the methodologies and insights drawn from the following research papers:

1. [*Love Me, Love Me Not: Human-Directed Sentiment Analysis in Arabic*](https://aclanthology.org/2022.nsurl-1.4.pdf)
Y Nahum, A Israeli, S Fine, K Bar - Proceedings of the Third International Workshop on NLP Solutions for Under Resourced Languages, 2022.
2. [*The idc system for sentiment classification and sarcasm detection in Arabic*](https://aclanthology.org/2021.wanlp-1.48/)
A Israeli, Y Nahum, S Fine, K Bar - Proceedings of the Sixth Arabic Natural Language Processing Workshop, 2021.

The codebase provided here is designed to closely follow the approaches and techniques discussed in these papers, offering a practical application and demonstration of their theoretical concepts in natural language processing and sentiment analysis.

## Usage example
### Install packages
```python
pip install AraEmotion
```
### Train TSDAE & MultiLabel Classification
```python
import pandas as pd
from araemotion.MultiLabelClassificationModel import EmotionMultilabelClassificationModel
train_df = pd.read_csv("datasets/train_6_labels.csv")
test_df = pd.read_csv("datasets/test_6_labels.csv")
labels = ['anger', 'disgust', 'fear', 'joy', 'sadness', 'neutral'] # the label names (column for each label - with the label as headers)

model = EmotionMultilabelClassificationModel(name_or_path="UBC-NLP/MARBERT",emotion_list=labels) # Init the model
model.train(train_df,test_df,num_epochs=6,TSDAE_pretrainig=False,save_model_dir="multilabel_6") # Train the model
```
### Evaluate the model performance
```python
model.evaluate(test_df)
```
### Predict emotion on new data
```python
from araemotion.MultiLabelClassificationModel import EmotionMultilabelClassificationModel
tweets = ['الناس ميتين جوع في ذمتكم الله لاسامحكم', 'فوق ماهو #أمان يجمع كل شعور حلو']
model = EmotionMultilabelClassificationModel(name_or_path="UBC-NLP/MARBERT",emotion_list=labels) # Init the model
model.predict(tweets) # Predict
```
### Train regular MultiLabel Classification model (no TSDAE)
```python
import pandas as pd
from araemotion.MultiLabelClassificationModel import EmotionMultilabelClassificationModel
train_12_labels = pd.read_csv("datasets/train_12_labels.csv")
test_12_labels = pd.read_csv("datasets/test_12_labels.csv")
labels = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love',
          'optimism', 'pessimism', 'sadness', 'surprise', 'trust', 'neutral']

model = EmotionMultilabelClassificationModel(name_or_path="UBC-NLP/MARBERT",emotion_list=labels) # Init the model
model.train(train_df,test_df,num_epochs=6,TSDAE_pretrainig=False,save_model_dir="multilabel_12") # Train the model
```

MIT License
```sql
Copyright (c)
2021 - 2022 Yotam Nahum

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
``
