from flask import Flask, render_template, request
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics.pairwise import cosine_similarity
import neattext.functions as ntfx
import numpy as np
import pandas as pd


def textcleaning(text):
    text = re.sub("'\'","", text)
    text = re.sub("[^a-zA-Z]", " ", text)
    text = ' '.join(text.split())
    text = text.lower()
    return text


file = open('Classifier/stopwords_english.pkl', 'rb')
stop_words = pickle.load(file)
file.close()


def removeStopwords(text):
    removedstopwords = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopwords)


def lemmatizing(sentence):
    lemma = WordNetLemmatizer()
    stemSentence = ''
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += ' '
    stemSentence = stemSentence.strip()
    return stemSentence


def stemming(sentence):
    stemmer = PorterStemmer()
    stemmed_sentence = ''
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemmed_sentence += stem
        stemmed_sentence += ' '

    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence


file1 = open('Classifier/genre_mlknn.pkl', 'rb')
model = pickle.load(file1)
file1.close()

file2 = open('Classifier/genre_tfidf.pkl', 'rb')
tfidf_vectorizer = pickle.load(file2)
file2.close()

file3 = open('Classifier/binarizer.pkl', 'rb')
multilabel_binarizer = pickle.load(file3)
file3.close()


def predict(text):
    text = textcleaning(text)
    text = removeStopwords(text)
    text = lemmatizing(text)
    text = stemming(text)

    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)
    return multilabel_binarizer.inverse_transform(predicted)


file4 = open('Recommender/books_df.pkl', 'rb')
books_df = pickle.load(file4)
file4.close()

file5 = open('Recommender/count_vect_matrix.pkl', 'rb')
count_vect = pickle.load(file5)
file5.close()

file6 = open('Recommender/count_vectorizer.pkl', 'rb')
count_vectorizer = pickle.load(file6)
file6.close()


app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template('index.html')


@app.route("/bookclassifier", methods=['GET', 'POST'])
def book_classifier():
    if request.method == 'POST':
        values = request.form['text']
        predicted = predict(values)
        if len(predicted[0]) == 0:
            result = 'Sorry, we are unable to predict which genre your book belongs to'
            return render_template('classifier.html', genre=result)
        else:
            return render_template('classifier.html', showresult=True, genre=predicted[0])
    return render_template('classifier.html')


@app.route("/recommender", methods=['GET', 'POST'])
def book_recommender():

    if request.method == 'POST':
        values = request.form['title']
        if len(values) == 0:
            return render_template('recommender.html')
        values = ntfx.remove_stopwords(values)
        values = ntfx.remove_special_characters(values)
        values = values.lower()
        cos_similarity_matrix = cosine_similarity(count_vect, count_vectorizer.transform([values]).astype(np.uint8))
        new_scores = pd.DataFrame(cos_similarity_matrix, columns=['scores'])
        n = 5
        n_largest = new_scores['scores'].nlargest(n + 1)
        book_index = [i for i in n_largest.index[1:]]
        book_score = [i for i in n_largest[1:]]
        recommended = books_df.iloc[book_index]
        recommended['Similarity_score'] = book_score

        final = recommended[
            ['ISBN', 'Book-Title', 'Similarity_score', 'Book-Author', 'Year-Of-Publication', 'Publisher',
             'Image-URL-L']]
        final.reset_index(inplace=True)
        final.drop('index', axis=1, inplace=True)
        final.index = final.index + 1
        final_list = final.values.tolist()
        titles = [final_list[i][1] for i in range(len(final_list))]
        isbn = [final_list[i][0] for i in range(len(final_list))]
        author = [final_list[i][3] for i in range(len(final_list))]
        year = [final_list[i][4] for i in range(len(final_list))]
        publisher = [final_list[i][5] for i in range(len(final_list))]
        first = titles[0]

        return render_template('recommender.html', showresult=True, titles=titles, isbn=isbn, author=author, year=year, publisher=publisher)
    return render_template('recommender.html')


if __name__ == '__main__':
    app.run()
