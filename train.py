import pandas
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from helper.helpers import normalize
from joblib import dump

comments = pandas.read_csv('./comments.csv', delimiter=",")

# Parse ideology
# Left wing -> 1
# Right wing -> 0
comments['ideology'] = (comments['ideology'] == 'left-wing').astype(int)

# Normalization
comments['text_original'] = comments['text_original'].apply(normalize)

# Download stopwords dictionaries
nltk.download("stopwords")
stopwords_loaded = stopwords.words('spanish')

# Remove stopwords
comments['text_original'] = comments['text_original'].apply(lambda text: ' '.join([word for word in text.split() if word not in stopwords_loaded]))

# Some messages only has emojis or stopwords, after normalize some observations can be empty
comments = comments.dropna(subset=['text_original'])

# Stemming
stemmer = SnowballStemmer('spanish')
comments['text_original'] = comments['text_original'].apply(lambda text: ' '.join([stemmer.stem(word) for word in text.split(' ')]))

nltk.download('wordnet')
nltk.download('omw-1.4')

# Lemmatizer
lemmatizer = WordNetLemmatizer()

comments['text_original'] = comments['text_original'].apply(lambda text: ' '.join([lemmatizer.lemmatize(word) for word in text.split(' ')]))

# Distribute observations
x_train, x_test, y_train, y_test = train_test_split(comments['text_original'].values, comments['ideology'].values, test_size=0.2)

# Pipeline
learning_stages = Pipeline([('frecuence', CountVectorizer()), ('tfidf', TfidfTransformer()), ('algorithm', MultinomialNB())])

model = learning_stages.fit(x_train, y_train)

report = classification_report(y_test, model.predict(x_test), digits=4)

predictions = model.predict(x_test)
for x, y, p in zip(x_test, y_test, predictions):
    if p == 0:
        continue
    print(x, y, p)

print(report)

dump(model, 'model/model.joblib')
