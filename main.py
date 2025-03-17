import os
import pickle
import re
from contextlib import asynccontextmanager
from enum import Enum

import spacy
import uvicorn
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer


class ModelType(Enum):
    naive_bayes = "nb"
    logistic_regression = "lr"
    random_forest = "rf"


class SentimentRequest(BaseModel):
    text: str
    model: ModelType


# Save the transformed TF-IDF vectorized data into pickle format for loading into REST API.
pickle_vectorizer_model_path = 'models/vectorized-model.pkl'

# Save the trained NB model into pickle format for loading into REST API.
pickle_nb_ml_model_path = 'models/nb-ml-model.pkl'

# Save the trained LR model into pickle format for loading into REST API.
pickle_lr_ml_model_path = 'models/lr-ml-model.pkl'

# Save the trained RF model into pickle format for loading into REST API.
pickle_rf_ml_model_path = 'models/rf-ml-model.pkl'

# Vectorized model for transforming input text to ML model required TF-IDF format.
vectorized_model: TfidfVectorizer

# The trained NB ML model.
nb_ml_model = None

# The trained LR ML model.
lr_ml_model = None

# The trained RF ML model.
rf_ml_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global vectorized_model
    global nb_ml_model

    print(f'[Service] - initializing ...')

    # Load the saved vectorized model.
    if os.path.exists(pickle_vectorizer_model_path):
        vectorized_model = pickle.load(open(pickle_vectorizer_model_path, 'rb'))
    else:
        print(f'[Service] - {pickle_vectorizer_model_path} not found!')

    # Load the saved NB model.
    if os.path.exists(pickle_nb_ml_model_path):
        nb_ml_model = pickle.load(open(pickle_nb_ml_model_path, 'rb'))
    else:
        print(f'[Service] - {pickle_nb_ml_model_path} not found!')

    # Load the saved LR model.
    if os.path.exists(pickle_lr_ml_model_path):
        lr_ml_model = pickle.load(open(pickle_lr_ml_model_path, 'rb'))
    else:
        print(f'[Service] - {pickle_lr_ml_model_path} not found!')

    # Load the saved RF model.
    if os.path.exists(pickle_rf_ml_model_path):
        rf_ml_model = pickle.load(open(pickle_rf_ml_model_path, 'rb'))
    else:
        print(f'[Service] - {pickle_rf_ml_model_path} not found!')

    yield
    vectorized_model = None
    nb_ml_model = None
    print(f'[Service] - shutting down ...')


app = FastAPI(lifespan=lifespan)


def preprocess_text(text):
    """
    The same method used in the notebook for preprocessing the raw text into TF-IDF vectorized format.
    :param text: Takes in a raw text.
    :return: TF-IDF vectorized data.
    """
    # Load spaCy English model
    nlp = spacy.load("en_core_web_sm")

    # Define spaCy's stop words
    stop_words = nlp.Defaults.stop_words

    # Contractions dictionary
    contractions = {
        "can't": "cannot", "won't": "will not", "i'm": "i am", "she's": "she is",
        "he's": "he is", "they're": "they are", "we're": "we are", "i've": "i have",
        "you're": "you are", "they've": "they have", "i'd": "i would", "we'd": "we would",
        "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
        "don't": "do not", "haven't": "have not", "omg": "oh my god",
        "aren't": "are not", "didn't": "did not", "doesn't": "does not", "hadn't": "had not",
        "hasn't": "has not", "isn't": "is not", "it's": "it is", "let's": "let us",
        "ma'am": "madam", "mightn't": "might not", "might've": "might have",
        "mustn't": "must not", "must've": "must have", "needn't": "need not",
        "o'clock": "of the clock", "shan't": "shall not", "she'd": "she would",
        "she'll": "she will", "that's": "that is", "there's": "there is",
        "there'd": "there would", "they'd": "they would", "they'll": "they will",
        "wasn't": "was not", "weren't": "were not", "what'll": "what will",
        "what're": "what are", "what's": "what is", "what've": "what have",
        "where's": "where is", "who'd": "who would", "who'll": "who will",
        "who're": "who are", "who's": "who is", "who've": "who have",
        "why's": "why is", "would've": "would have", "you'd": "you would",
        "you'll": "you will", "you've": "you have", "y'all": "you all"
    }

    # Emojis dictionary
    emojis_dict = {
        ':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
        ':-(': 'sad', ':-<': 'sad', ':-P': 'raspberry', ':O': 'surprised',
        ':-@': 'shocked', ':@': 'shocked', ':-$': 'confused', r':\\': 'annoyed',
        ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
        '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.O': 'confused',
        '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ':-)': 'sadsmile', ';)': 'wink',
        'O:-)': 'angel', 'O*)': 'angel', '(:-D': 'gossip', '=^.^=': 'cat'
    }

    # 1️⃣ Text Cleaning
    text = re.sub(r'https?:\/\/[\S]+', '', text)  # Remove hyperlinks
    text = re.sub(r'[#,.\-$!/()?%_√Ø¬ø¬Ω:&|;]', ' ', text)  # Replace special characters with a space
    text = re.sub(r'@[A-Za-z0-9_]+', '', text)  # Remove mentions
    text = re.sub(r'\d{1,4}[\/\-]\d{1,2}[\/\-]\d{1,4}', '', text)  # Remove dates (e.g. 12/24/03)
    text = re.sub(r'\d+', '', text)  # Remove standalone numbers
    text = re.sub(r'\bbr\b', '', text)  # Remove noise "br"

    # 2️⃣ Contraction Expansion
    for contraction, full_form in contractions.items():
        text = re.sub(r'\b' + re.escape(contraction) + r'\b', full_form, text, flags=re.IGNORECASE)

    # 3️⃣ Emoji Replacement
    for emoji, meaning in emojis_dict.items():
        text = text.replace(emoji, f' {meaning} ')

    # 4️⃣ Normalisation: convert to lowercase and remove extra spaces
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    # 5️⃣ Tokenization, Lemmatisation, and Stopword Removal using spaCy
    doc = nlp(text)
    lemmatized_words = [token.lemma_ for token in doc]
    filtered_words = [word for word in lemmatized_words if word not in stop_words and word.isalpha()]

    return " ".join(filtered_words)


def predict(text: str, model):
    print(f'[Service] - preprocessing ...')
    processed_text = preprocess_text(text)

    print(f'[Service] - transforming data to TD-IDF ...')
    transformed_data = vectorized_model.transform([processed_text])

    result = model.predict(transformed_data)
    if result == 1:
        return "positive"
    else:
        return "negative"


@app.post("/predict")
def predict(request: SentimentRequest):
    try:
        text = request.text
        model = request.model

        if model == ModelType.naive_bayes:
            return predict(text, nb_ml_model)
        elif model == ModelType.logistic_regression:
            return predict(text, lr_ml_model)
        elif model == ModelType.random_forest:
            return predict(text, rf_ml_model)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported model')
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f'error: {e}')


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
