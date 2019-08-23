import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import spacy
import alibi
from alibi.datasets import fetch_movie_sentiment
from alibi.utils.download import spacy_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.pipeline import Pipeline
import dill
import joblib

# load data
movies = fetch_movie_sentiment()
movies.keys()
data = movies.data
labels = movies.target
target_names = movies.target_names

# define train and test set
np.random.seed(0)
train, test, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=42)
train, val, train_labels, val_labels = train_test_split(train, train_labels, test_size=.1, random_state=42)
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)
val_labels = np.array(val_labels)

# define and  train an cnn model
vectorizer = CountVectorizer(min_df=1)
clf = LogisticRegression(solver='liblinear')
pipeline = Pipeline([('preprocess', vectorizer), ('clf', clf)])

print('Training ...')
pipeline.fit(train, train_labels)
print('Training done!')

print("Creating an explainer")
predict_fn = lambda x: pipeline.predict(x)

model = 'en_core_web_md'
spacy_model(model=model)
nlp = spacy.load(model)

explainer = alibi.explainers.AnchorText(nlp, predict_fn)

explainer.predict_fn = None  # Clear explainer predict_fn as its a lambda and will be reset when loaded
print('Saving explainer')
with open("explainer.dill", 'wb') as f:
    dill.dump(explainer, f)

print("Saving individual files")
# Dump files - for testing creating an AnchorExplainer from components
joblib.dump(pipeline, "model.joblib")
joblib.dump(train, "train.joblib")
