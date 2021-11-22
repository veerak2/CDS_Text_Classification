from CDS_Classifier import train,data
from config import config
import pickle
from pathlib import Path

VECTORIZER_PATH = Path(config.STORE_DIR,"vectorizer")

def prediction(text: str):

    model = train.train()

    text = data.preprocess_text(text,stop_words=config.STOPWORDS,remove_punctuations=True, remove_numbers=True,stem=True)

    with open(Path(VECTORIZER_PATH,"tfidf.txt"),'rb') as f:
        tfidf_transformer = pickle.load(f)

    text = tfidf_transformer.transform([text])
    
    return model.predict(text)