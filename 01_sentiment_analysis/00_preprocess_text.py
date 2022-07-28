import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.corpora.dictionary import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

import pandas as pd
import numpy as np

import pickle

from bs4 import BeautifulSoup

from spellchecker import SpellChecker

df_train = pd.read_csv("../data/sentiment_analysis/train.csv")
df_test = pd.read_csv("../data/sentiment_analysis/test.csv")

# Preprocess
df_train["istrain"] = True
df_test["istrain"] = False

df = pd.concat([df_train, df_test], axis=0)


def remove_url(text):
    url = re.compile(r'https?://\S+|www\.\S+')
    return url.sub(r'',text)


df['text'] = df['text'].apply(lambda x: remove_url(x))


def lower_text_and_remove_special_chars(text):
    text = text.lower().strip()
    return re.sub(r"\W+", " ", text)


df['text'] = df['text'].apply(lambda x: lower_text_and_remove_special_chars(x))


def remove_html_tags(text):
    return BeautifulSoup(text, features="html.parser").get_text()


df['text'] = df['text'].apply(lambda x: remove_html_tags(x))


def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)


df['text'] = df['text'].apply(lambda x: remove_emoji(x))


def fix_contractions(text):
    # performing de-contraction
    text = text.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", "is") \
        .replace("'m", "am").replace("'t", "not") \
        .replace("'ll", " will")

    return text


df['text'] = df['text'].apply(lambda x: fix_contractions(x))


train = df.query("istrain==True")
test = df.query("istrain==False")

tokenizer = Tokenizer()
tokenizer.oov_token = '<oovToken>'
tokenizer.fit_on_texts(train.text)
vocab = tokenizer.word_index
vocab_count = len(vocab)+1

xTrain = pad_sequences(tokenizer.texts_to_sequences(train.text.to_numpy()), padding='pre', maxlen=60)
yTrain = train.target.to_numpy()

xTest = pad_sequences(tokenizer.texts_to_sequences(test.text.to_numpy()), padding='pre', maxlen=60)

train_dict = {'x_train': xTrain, 'y_train': yTrain, 'x_test': xTest, 'vc': vocab_count, 'vocab': vocab}

with open('../data/sentiment_analysis/train_data.dict', 'wb') as f:
    pickle.dump(train_dict, f)




