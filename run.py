import nltk
from nltk.corpus import stopwords
from helper.helpers import normalize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from joblib import load

model = load('model/model.joblib')

nltk.download("stopwords")
nltk.download('wordnet')
nltk.download('omw-1.4')

stopwords_loaded = stopwords.words('spanish')

print("Input phrase \n")
phrase = input()

phrase = normalize(phrase)
phrase = ' '.join([word for word in phrase.split() if word not in stopwords_loaded])
phrase = ' '.join([SnowballStemmer('spanish').stem(word) for word in phrase.split(' ')])
phrase = ' '.join([WordNetLemmatizer().lemmatize(word) for word in phrase.split(' ')])

prediction = model.predict([phrase])

if prediction[0] == 0:
    print("Es de derechas")
else:
    print("Es de izquierdas")
