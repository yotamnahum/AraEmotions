# Emotion analysis on arabic tweets
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