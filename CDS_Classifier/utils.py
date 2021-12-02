import tensorflow as tf
from gensim.models import Word2Vec
from pathlib import Path

def deleteContent(fName):
    with open(fName, "w"):
        pass

def clear_keras_backend():
    tf.keras.backend.clear_session()

def get_word2vec_model(path):
    
    return Word2Vec.load(str(path / 'custom_word2vec.model'))