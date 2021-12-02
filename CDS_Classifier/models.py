from re import T
from sklearn.ensemble import GradientBoostingClassifier

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, TextVectorization, GlobalAveragePooling1D
from tensorflow.keras.layers import Activation, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import losses
from . import tf_layers
from pathlib import Path
from config import config
from CDS_Classifier import utils,data

def gb_classifier():
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
                                        max_depth=1, random_state=0)
    return model


#getword2vec
print(Path(config.STORE_DIR,'word2vec'))
word2vec = utils.get_word2vec_model(Path(config.STORE_DIR,'word2vec'))
print('loaded WORD2VEC')

vocabs = word2vec.wv.index_to_key

embedding_matrix = data.create_embedding_matrix(word2vec)
print('created EMBEDDING MATRIX')

def nn_simple_classifier():

    model = Sequential([
        tf_layers.get_keras_vectorization_layer(vocabs=vocabs),
        tf_layers.get_embedding_layer(embedding_matrix=embedding_matrix), 
        Dropout(0.2),
        GlobalAveragePooling1D(),
        Dropout(0.2),
        Dense(1)])

    return model