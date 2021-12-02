from CDS_Classifier import data,models,utils
from config import config
from pathlib import Path
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.keras import losses


def train():

    #data path
    data_path = Path(config.DATA_DIR, "cleaned_dataset_no_personal_info.csv")
    #load data
    df = pd.read_csv(data_path)

    df.dropna(subset=['Valid 0/1_x'], inplace=True)
    df['transcript'] = df['transcript'].apply(data.preprocess_text,stop_words=config.STOPWORDS,remove_punctuations=True, remove_numbers=True,stem=True)
    #splitting the data

    

    X_train, X_val, X_test, y_train, y_val, y_test = data.stratified_train_val_test_split(df)

    X_train.drop(['Course'],axis=1,inplace=True)
    X_val.drop(['Course'],axis=1,inplace=True)
    X_test.drop(['Course'],axis=1,inplace=True)

    #vectorize the data
    #X_train, X_val, X_test = data.vectorizer(list(X_train['transcript']),list(X_val['transcript']),list(X_test['transcript']))
    
    model = models.nn_simple_classifier()

    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
             optimizer = 'adam',
             metrics = tf.metrics.BinaryAccuracy(threshold=0.0))

    model.fit(
        X_train,y_train,validation_data=(X_val,y_val), epochs=5)    

    return model

  