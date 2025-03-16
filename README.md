# IMDB Sentiment Analysis
A jupyter notebook project for building and deploying ML models for IMDB sentiment analysis. The following supervised \
ML models are built.
- Naive Bayes
- Logistic Regression
- Random Forest

## How to run?
1. Clone the project to local: git clone: `git@github.com:everhettoo/imdb-sentiment-analysis.git`
2. Open the project using preferred IDE (PyCharm or VSCode)
3. Create a virtual environment (if needed): `python -m venv venv `
4. Install required packages: `pip install -r requirements.txt`

## Project structure (explained)
1. `1.0-exploratory-data-analysis.ipynb` - EDA & Preprocessing that creates `data/2-imdb-movie-review-processed.csv` for model building
2. `2.1-model-multinomial-naive-bayes.ipynb` - Naive Bayes model built using preprocessed dataset `data/2-imdb-movie-review-processed.csv`
3. `2.2-model-logistic-regression.ipynb` - Logistic Regression model built using preprocessed dataset `data/2-imdb-movie-review-processed.csv`
4. `2.3-model-random-forest.ipynb` - Random Forest model built using preprocessed dataset `data/2-imdb-movie-review-processed.csv`