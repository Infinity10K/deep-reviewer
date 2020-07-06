from django.shortcuts import render
from django.http import HttpResponse

from tensorflow import keras
import numpy as np
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df = pd.read_csv('./models/imdb_master.zip', encoding="latin-1")

df = df[df.type == 'train']
df.drop(['type'], axis = 1, inplace=True)
df = df.drop(['Unnamed: 0'], axis=1)
df = df[df.label != 'unsup']
df = df.drop(['file'],axis=1)

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

df['review'] = df.review.apply(lambda x: clean_text(x))

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['review'])

maxlen = 370

model = keras.models.load_model('./models/sentem_model.h5')
model2 = keras.models.load_model('./models/mark_model.h5')

def home(request):
    return render(request, 'index.html')


def predict(request):
    temp = []
    temp.append(request.POST.get('text'))

    list_sentences_test = temp
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    sentem = model.predict(X_te)[0,0]
    mark = model2.predict(X_te)

    context = {'sentem': sentem, 'mark': np.argmax(mark[0,:])+1, 'review': temp[0]}

    return render(request, 'index.html', context)