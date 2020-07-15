from django.shortcuts import render

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

df = pd.read_csv('./models/imdb_clean.zip')

max_features = 6000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(df['review'])

maxlen = 370

model = load_model('./models/sentem_model.h5')
model2 = load_model('./models/mark_model.h5')

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


def home(request):
    return render(request, 'index.html')


def predict(request):
    temp = [request.POST.get('text')]
    test = [clean_text(temp[0])]
    list_sentences_test = test
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)
    sentem = model.predict(X_te)
    mark = model2.predict(X_te)

    context = {'sentem': sentem[0,0], 'mark': np.argmax(mark[0,:])+1, 'review': temp[0]}

    return render(request, 'index.html', context)
