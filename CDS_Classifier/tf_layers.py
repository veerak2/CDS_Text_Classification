from pathlib import Path
from tensorflow.keras.layers import Embedding, TextVectorization
import numpy as np


def get_keras_vectorization_layer(vocabs: list = None, max_tokens: int = None) -> TextVectorization:
    
    if vocabs:
        text_vectorization = TextVectorization(
            max_tokens=max_tokens, standardize='lower_and_strip_punctuation',
            split='whitespace', ngrams=None, output_mode='int',
            output_sequence_length=None, pad_to_max_tokens=False, vocabulary=vocabs
        )
        print('Using vocab from pretrained words')
        print(f'number of vocabulary is {len(vocabs)}')
        
    else:
        text_vectorization = TextVectorization(
            max_tokens=max_tokens, standardize='lower_and_strip_punctuation',
            split='whitespace', ngrams=None, output_mode='int',
            output_sequence_length=None, pad_to_max_tokens=False
        )

    return text_vectorization

def get_embedding_layer(input_dim: int = None, output_dim: int = None, embedding_matrix: np.array = None, trainable: bool = False, mask_zero: bool = True):
   
    if embedding_matrix.shape:
        embedding_layer = Embedding(input_dim=embedding_matrix.shape[0]+2, output_dim=embedding_matrix.shape[1], weights = [embedding_matrix], trainable = trainable, mask_zero=mask_zero)
        print(f'shape of pretrained embedding matrix: {embedding_matrix.shape}')
    else:
        embedding_layer = Embedding(input_dim=input_dim, output_dim=output_dim, trainable = trainable, mask_zero=mask_zero)
        
    return embedding_layer