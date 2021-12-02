import re,pickle
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from pathlib import Path
from config.config import CONFIG_DIR, STORE_DIR
from CDS_Classifier import utils

VECTORIZER_PATH = Path(STORE_DIR,"vectorizer")

def preprocess_text(text, stop_words: list, remove_numbers: True, remove_punctuations: True, stem: False,lemma: bool = False) -> str:       
    
    if remove_numbers:
        text = re.sub('\d','',text) #removes any numbers
        text = re.sub(' +',' ',text)

    if len(stop_words):
        
        text = [word for word in text.split(' ') if not word in stop_words]
        text = " ".join(text)
    
    if remove_punctuations:
        text = re.sub(r'[^\w\s]','',text)
        text = re.sub(' +', ' ',text)
    
    if stem:
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text.split(' ')]
        text = " ".join(text)
    
    if lemma:
        pass
        
    text = re.sub(' +',' ',text) #remove extra white spaces
    text = text.strip()
    
    return text

def stratified_train_val_test_split(data, train_size=0.6):
    X = data[['transcript','Course']]
    y = data['Valid 0/1_x']
    stratify = 'Course'
    X_train, X_, y_train, y_ = train_test_split(X,y,shuffle =True,train_size=train_size)
    X_val, X_test, y_val, y_test = train_test_split(X_,y_,shuffle =True,train_size = 0.5)
    return X_train, X_val, X_test, y_train, y_val, y_test

def vectorizer(train,val,test):
    tfidf = TfidfVectorizer()
    train_fit = tfidf.fit(train)
    train = tfidf.transform(train)
    val = tfidf.transform(val)
    test = tfidf.transform(test)
    print(STORE_DIR)
    utils.deleteContent(Path(VECTORIZER_PATH,"tfidf.txt"))
    with open(Path(VECTORIZER_PATH,"tfidf.txt"),'wb') as f:
        pickle.dump(train_fit,f,pickle.HIGHEST_PROTOCOL)
    return train,val,test

def create_embedding_matrix(word2vec_model: Word2Vec) -> np.array:
    
    embedding_matrix = np.zeros((len(word2vec_model.wv),word2vec_model.wv.vector_size))
    print(f'vocab length = {len(word2vec_model.wv)}')
    print(f'vector length = {word2vec_model.wv.vector_size}')
    for i in range(len(word2vec_model.wv)):
        embedding_vector = word2vec_model.wv[word2vec_model.wv.index_to_key[i]]
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix


