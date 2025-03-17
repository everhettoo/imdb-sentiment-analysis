from enum import Enum
import re
import uvicorn
from fastapi import FastAPI, status, HTTPException
import pickle
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer

from pydantic import BaseModel


class ModelType(Enum):
    naive_bayes = "nb"
    logistic_regression = "lr"
    random_forest = "rf"


class SentimentRequest(BaseModel):
    text: str
    model: ModelType


app = FastAPI()


def preprocess_text(text):
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

    # If not None, build a vocabulary that only consider the top max_features ordered by term frequency across the corpus. Otherwise, all features are used.
    # Limits the max words used for building the model.
    max_features = 20000

    # TODO: Afif, please confirm and justify!
    # Initialize TF-IDF Vectorizer
    # vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features)
    vectoriser = TfidfVectorizer(ngram_range=(1, 2))

    # Fit TF-IDF on cleaned training data
    vectoriser.fit(filtered_words)

    transformed_data = vectoriser.transform([" ".join(filtered_words)])

    return transformed_data


def nb_predict(text: str):
    model = pickle.load(open("models/nb-model.pkl", "rb"))

    # array.reshape(1, -1)
    result = 0
    try:
        processed_text = preprocess_text(text)
        result = model.predict(processed_text)
    except Exception as exc:
        print(exc)

    return result


@app.post("/predict")
def predict(request: SentimentRequest):
    # loaded_nb = pickle.load(open("models/nb-model.pkl", "rb"))
    # result = loaded_nb.score(X_test, y_test)
    # return data
    text = request.text
    model = request.model

    if model == ModelType.naive_bayes:
        return nb_predict(text)
    elif model == ModelType.logistic_regression:
        pass
    elif model == ModelType.random_forest:
        pass
    else:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail='Unsupported model')

    return {text: model}


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
