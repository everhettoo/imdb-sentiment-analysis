import re

url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
contractions = {
    "can't": "cannot", "won't": "will not", "i'm": "i am", "she's": "she is",
    "he's": "he is", "they're": "they are", "we're": "we are", "i've": "i have",
    "you're": "you are", "they've": "they have", "i'd": "i would", "we'd": "we would",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
    "don't": "do not", "doen't": "does not", "haven't": "have not", "omg": "oh my god",
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


def remove_url(text: str) -> str:
    return re.sub(url_pattern, '', text)


def remove_html(text: str) -> str:
    return re.sub(r'<.*?>', '', text)


def remove_bracketed_text(text: str) -> str:
    return re.sub(r'\(.*?\)', '', text)


def remove_extra_space(text: str) -> str:
    return re.sub(' +', ' ', text)


def transform_contractions(text: str) -> str:
    words = text.lower().split()
    for word in words:
        if word in contractions:
            text = text.replace(word, contractions[word])

    return text


def get_words(text: str) -> str:
    return re.sub(r'[^a-z]', ' ', text)
