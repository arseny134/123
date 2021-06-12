from text_processing import *

cur_dir = os.path.dirname(__file__)

tf1 = pickle.load(open(
    os.path.join(cur_dir,
                 'pkl',
                 'tf_vocab.pkl'), 'rb'))


def file_pandas_proc(text):  # обработка текста помещенного в Датафрейм

    text = text.apply(remove_stop_words)
    text = text.apply(lemmatize_text)
    text = text.apply(remove_stop_words)
    text = text.apply(ClearData)

    return text


def Tfidf_File(text):  # векторизация
    tf1_new = TfidfVectorizer(strip_accents=None,
                              lowercase=False,
                              preprocessor=None, vocabulary=tf1.vocabulary_)

    text = tf1_new.fit_transform(text)

    return text
