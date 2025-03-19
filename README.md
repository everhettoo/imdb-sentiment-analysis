# IMDB Sentiment Analysis

A project for building and deploying ML models for IMDB sentiment analysis. \
The following supervised ML models are built.

- Naive Bayes
- Logistic Regression
- Random Forest

## There are two parts to this project:

1. Jupyter - notebooks for preprocessing and building the ML models.
2. REST-API - the `main.py` hosts the models as REST endpoint (FastAPI).

## How to run?

1. Clone the project to local: git clone: `git@github.com:everhettoo/imdb-sentiment-analysis.git`
2. Open the project using preferred IDE (PyCharm or VSCode)
3. Create a virtual environment (if needed): `python -m venv venv `
4. Install required packages: `pip install -r requirements.txt`
5. If need to run main.py before running any notebooks: `python -m spacy download en_core_web_sm`

## Project structure (explained)

1. `1.0-exploratory-data-analysis.ipynb` - EDA & Preprocessing that creates `data/2-imdb-movie-review-processed.csv` for
   model building
2. `2.1-model-multinomial-naive-bayes.ipynb` - Naive Bayes model built using preprocessed dataset
   `data/2-imdb-movie-review-processed.csv`
3. `2.2-model-logistic-regression.ipynb` - Logistic Regression model built using preprocessed dataset
   `data/2-imdb-movie-review-processed.csv`
4. `2.3-model-random-forest.ipynb` - Random Forest model built using preprocessed dataset
   `data/2-imdb-movie-review-processed.csv`
5. `3.0-main.py` - Loads the trained model (in pickle format) and host a REST endpoint at: http://localhost:8000/docs
6. `data` - folder where the original and processed data are stored
7. `models` - folder where all the trained models are stored