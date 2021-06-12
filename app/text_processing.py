import re
import os
import pickle
from nltk import word_tokenize
from pymystem3 import Mystem
from sklearn.feature_extraction.text import TfidfVectorizer

cur_dir = os.path.dirname(__file__)

# загружаем список стоп-слов
stopwords = pickle.load(open(
    os.path.join(cur_dir,
                 'pkl',
                 'stopwords.pkl'), 'rb'))

# загружаем сохраненный словарь векторайзера, полученный при обработке обучающей выборки
tf1 = pickle.load(open(
    os.path.join(cur_dir,
                 'pkl',
                 'tf_vocab.pkl'), 'rb'))


# удаление стоп-слов
def remove_stop_words(text):
    tokens = word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in stopwords and token != ' ']
    return " ".join(tokens)


# лемматизация текста
mystem = Mystem()


def lemmatize_text(text):
    text_lem = mystem.lemmatize(text)
    tokens = [token for token in text_lem if token != ' ']
    return " ".join(tokens)


# очистка текста
def ClearData(text):
    # удаляем всю html разметку из текста
    text = re.sub('(?:<).*?(?:>)|([0-9]+)|([a-z]+)', '', text)
    # удаляем все слова , длинна которых меньше 3-х символов
    text = ' '.join(word for word in text.split() if len(word) > 3)
    # удаляем все не словарные символы
    text = (re.sub('[\W]+', ' ', text))
    return text


# векторизация
def Tfidf(text):
    tf1_new = TfidfVectorizer(strip_accents=None,  # создаем векторайзер
                              lowercase=False,
                              preprocessor=None,
                              vocabulary=tf1.vocabulary_)  # и сохраняем в нем словарь из старого векторайзера
    text = [text]
    text = tf1_new.fit_transform(text)

    return text
