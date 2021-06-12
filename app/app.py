import pandas as pd
import os
from text_processing import *
from file_processing import *
from flask import Flask, flash, render_template, request, redirect
from wtforms import Form, TextAreaField, validators, SelectField
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'D:/clallif/uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv', 'docx'}
# MAX_CONTENT_LENGTH = 16 * 1000 * 1000

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///texts_base.db'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

######## загрузка предобученной модели
cur_dir = os.path.dirname(__file__)
classificator = pickle.load(open(os.path.join(cur_dir,
                                              'pkl',
                                              'sgdc.pkl'), 'rb'))

SECRET_KEY = os.urandom(32)
app.config['SECRET_KEY'] = SECRET_KEY


class Base(db.Model):  # создаем дб
    id = db.Column(db.Integer, primary_key=True)
    texts = db.Column(db.Text, nullable=False)
    topics = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return '<Base %r>' % self.id


def classify(text):  # обработка текста и классификация его моделью
    text = remove_stop_words(lemmatize_text(remove_stop_words(ClearData((text)))))
    text = Tfidf(text)
    pred = classificator.predict(text)

    return pred


def train(text, pred):
    text = remove_stop_words(lemmatize_text(remove_stop_words(ClearData((text)))))
    text = Tfidf(text)
    classificator.partial_fit(text, [pred])  # метод для дообучения модели на новых данных


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


######## Flask
class TextForm(Form):  # текстовая форма на главной странице
    text_form = TextAreaField('',
                              [validators.DataRequired(),
                               validators.length(min=10)],
                              render_kw={"placeholder": "Введите текст для определения категории"})


class Select(Form):  # выбор категории в бд и на странице результата
    select = SelectField('',
                         choices=['Авто и ГИБДД', 'Авторское право', 'Адвокаты и юристы',
                                  'Административный кодекс', 'Армия. Военнное право',
                                  'Банки. Кредит. Страховка', 'Банкротство. Коллекторы',
                                  'Бизнес и организации', 'Гражданский кодекс', 'Гражданство и миграция',
                                  'Дачи, земля, межевание', 'Документы. Бланки',
                                  'ЖКХ. Коммунальные услуги', 'Законы и Кодексы', 'Материнство и детство',
                                  'Медицина', 'Налоги и споры', 'Недвижимость и прописка',
                                  'Нотариус и наследство', 'Образование', 'Охота, оружие, браконьеры',
                                  'Пенсия. Ветераны', 'Персональные данные', 'Почта', 'Семейное право',
                                  'Служба судебных приставов', 'Соц.зашита', 'Соц.сети, игры и интернет',
                                  'Суд', 'Торговля и ЗПП', 'Трудовые отношения', 'Туризм и заграница',
                                  'Уголовный кодекс'])


@app.route('/')
def index():
    form = TextForm(request.form)
    return render_template('textsform.html', form=form)


@app.route('/', methods=['POST', 'GET'])
def load_file():
    if request.method == 'POST':

        if 'file' not in request.files:  # проверки файла на допустимый формат и тд
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']

        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            text = file.read().decode("utf-8-sig")
            text = text.splitlines()  # обработка файла
            df = pd.DataFrame(columns=['texts'])  # создание объекта датафрейм для хранения файла
            df['texts'] = (text)
            dfbase = pd.DataFrame(columns=['texts', 'preds_topics'])
            dfbase['texts'] = df['texts']

            text = file_pandas_proc(df['texts'])
            text = Tfidf_File(text)  # векторизация  файла
            pred = classificator.predict(text)  # определение категории текстов внутри файла
            dfbase['preds_topics'] = pred

            # dfbase.to_csv('data5TEST.csv', encoding='utf-8-sig')
            dfbase.to_excel('data.xlsx', encoding='utf-8-sig')  # сохранение в файл с определенными категориями

            for i, j in zip(dfbase['texts'], dfbase['preds_topics']):  # сохранение в бд
                base = Base(texts=i, topics=j)
                db.session.add(base)
                db.session.commit()

            return redirect('/')


@app.route('/results', methods=['POST'])
def results():  # отправка данных из текстовой формы на страницу результата
    form = TextForm(request.form)
    form2 = Select(request.form)
    if request.method == 'POST' and form.validate():
        data = request.form['text_form']
        pred = classify(data)  # определение категории текста
        return render_template('results.html',
                               content=data,
                               prediction=pred[0], form=form2)

    return render_template('textsform.html', form=form)


@app.route('/database', methods=['POST', 'GET'])
def database():
    base = Base.query.order_by(Base.id.desc()).all()  # показать все записи в бд с сортировкой от новых к старым
    form2 = Select(request.form)
    form3 = Select(request.form)  # фильтр категорий в бд

    # на странице /result
    if request.method == "POST":

        feedback = request.form['feedback_button']

        data = request.form['data']  # текст = содержимому текстовой формы
        topics = form2.select.data  # если не нажали кнопку верно, то категория равна выбранной категории из списка

        if feedback == 'Верно':  # если нажали кнопку верно, то категория равна той которую определила модель
            topics = request.form['prediction']

        train(data, topics)  # дообучение модели на новом тексте
        base2 = Base(texts=data, topics=topics)  # сохранение в бд
        db.session.add(base2)
        db.session.commit()
        base = Base.query.order_by(Base.id.desc()).all()

    return render_template('database.html', base=base, form3=form3)


@app.route('/insert_db', methods=['POST', 'GET'])
def insert_db():  # вставка записей в бд вручную
    if request.method == "POST":
        texts = request.form['texts']
        topics = request.form['topics']
        train(texts, topics)
        base = Base(texts=texts, topics=topics)

        try:
            db.session.add(base)
            db.session.commit()
            return redirect('/database')
        except:
            return 'Ошибка при добавлении в бд'

    return render_template('insert_db.html')


@app.route('/database/<int:id>', methods=['POST', 'GET'])
def delete_one_texts(id):
    base = Base.query.get_or_404(id)  # получаем id  и удаляем тексты по одному из бд
    try:
        db.session.delete(base)
        db.session.commit()
        return redirect('/database')
    except:
        return "При удалении произошла ошибка"


@app.route('/database/delete_all_texts', methods=['POST', 'GET'])
def delete_all_texts():  # удаление всех текстов из бд
    db.session.query(Base).delete()
    db.session.commit()
    return redirect('/database')


@app.route('/database/topic', methods=['POST', 'GET'])
def topics_db():
    form3 = Select(request.form)  # фильтр категорий в бд
    if request.method == "POST":
        filters = request.form['filter_button']
        if filters == 'Выбрать':
            topic = form3.select.data

            base = Base.query.filter_by(topics=topic)  # выдать только ту категорию которая установлена в фильтре
            return render_template('database.html', base=base, form3=form3)


if __name__ == '__main__':
    app.run(debug=True)
