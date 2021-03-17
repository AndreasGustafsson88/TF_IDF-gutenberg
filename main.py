import re
import math
import os
import requests
import nltk
import numpy as np

from collections import defaultdict, Counter

from bs4 import BeautifulSoup

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

from num2words import num2words


# EXAMPLE 1. Compare two books
def get_books(links):
    books = []

    for link in links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')

        # content of book
        story = [p.get_text() for p in soup.select("p")]

        books.append(str(story))

    return books, len(books)


# Preprocess
def remove_symbols(data):
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~"
    for i in range(len(symbols)):
        data = np.char.replace(data, symbols[i], ' ')
        data = np.char.replace(data, "  ", " ")
    data = np.char.replace(data, ',', '')
    return data


def convert_lower_case(data):
    return np.char.lower(data)


def remove_apostrophe(data):
    return np.char.replace(data, "'", "")


def remove_stop_words(data):
    stop_words = stopwords.words('english')
    words = word_tokenize(str(data))

    new_text = ""
    for w in words:
        if w not in stop_words:
            new_text = new_text + " " + w
    return new_text


def stemming(data):
    stemmer = PorterStemmer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + stemmer.stem(w)
    return new_text


def convert_numbers(data):
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        try:
            w = num2words(int(w))
        except:
            a = 0
        new_text = new_text + " " + w
    new_text = np.char.replace(new_text, "-", " ")
    return new_text


def remove_single_char(data):
    return [word for word in str(data).split() if len(word) > 1]


def lemmetizer(data):
    lemmer = WordNetLemmatizer()
    tokens = word_tokenize(str(data))
    new_text = ""
    for w in tokens:
        new_text = new_text + " " + lemmer.lemmatize(w)
    return new_text


def remove_linebreaks(data):
    clean_data = list(map((lambda x: x.replace('\\n', ' ').replace('\\r', ' ')), data.split()))
    return " ".join(clean_data)


def preprocess(data, n, i):
    print(f"book {i}/{n} was {len(data.split())} words long")

    data = remove_linebreaks(data)
    data = convert_lower_case(data)
    data = remove_symbols(data)
    data = remove_stop_words(data)
    data = remove_apostrophe(data)
    data = convert_numbers(data)
    data = lemmetizer(data)
    data = stemming(data)
    data = remove_symbols(data)
    data = remove_single_char(data)

    print(f"book is now {len(data)} words long")
    print('=' * 20)

    return data


def calculate_tf(documents, n, show=True):
    word_dict = defaultdict(dict)
    for book_index, document in enumerate(documents):
        for word in document:
            if word not in word_dict:
                book_indexes = {i: [0] for i in range(n)}
                word_dict[word] = book_indexes
                word_dict[word]['DF'] = 0
            if word_dict[word][book_index][0] == 0:
                word_dict[word][book_index] = [document.count(word) / len(document)]
                word_dict[word]['DF'] += 1
    if show:
        [print(word, word_dict[word]) for word in list(word_dict.keys())[3:8]]
    return word_dict


def calculate_tf_idf(tf_document, n, show=True):
    for word, book_indexes in tf_document.items():
        for book_index, tf in book_indexes.items():
            if book_index != 'DF':
                tf_document[word][book_index].append(tf[0] * (math.log10(n / tf_document[word]['DF']) + 1))
    if show:
        [print(word, tf_document[word]) for word in list(tf_document.keys())[3:8]]
    return tf_document


def vectorize_tf_idf(tf_idf_document, n):
    vector_doc = [[] for _ in range(n)]
    [vector_doc[i].append(tf_idf_document[word][i][1]) for i in range(n) for word in tf_idf_document]
    return vector_doc


def dot(v1, v2):
    return sum(v * u for v, u in zip(v1, v2))


def cosine_calculation(u, v):
    return dot(u, v) / ((dot(u, u)**0.5) * (dot(v, v)**0.5))


def cosine_comparison(comparison_doc, v_doc):
    return [cosine_calculation(comparison_doc, doc) for doc in v_doc]


def get_all_books(book_to_compare):
    regex = "(?<=of )(.*)(?=, by)"
    books, titles = [], []
    links = [
        'https://www.gutenberg.org/files/30667/30667-h/30667-h.htm',
        "https://www.gutenberg.org/files/13670/13670-h/13670-h.htm",
        "https://www.gutenberg.org/files/7423/7423-h/7423-h.htm",
        "https://www.gutenberg.org/files/6768/6768-h/6768-h.htm",
        "https://www.gutenberg.org/files/4682/4682-h/4682-h.htm",
        "https://www.gutenberg.org/files/3829/3829-h/3829-h.htm",
        "https://www.gutenberg.org/files/16921/16921-h/16921-h.htm",
        "https://www.gutenberg.org/files/25550/25550-h/25550-h.htm",
        "https://www.gutenberg.org/files/31619/31619-h/31619-h.htm",
        "https://www.gutenberg.org/files/84/84-h/84-h.htm",
    ]

    for link in links:
        page = requests.get(link)
        soup = BeautifulSoup(page.content, 'html.parser')
        # book title
        title = soup.find('title')
        fixed_title = re.search(regex, title.string)[0]
        # book content
        story = [p.get_text() for p in soup.select("p")]

        books.append(str(story))
        titles.append(str(fixed_title))

    # Get compared book content
    page = requests.get(book_to_compare)
    soup = BeautifulSoup(page.content, 'html.parser')

    title = 'Book_to_compare'
    story = [p.get_text() for p in soup.select("p")]

    books.append(str(story)), titles.append(title)

    return books, len(books), titles


def main():
    # EXAMPLE 1, Compare two books

    books_to_download = [
        'https://www.gutenberg.org/files/30667/30667-h/30667-h.htm',
        'https://www.gutenberg.org/files/7766/7766-h/7766-h.htm'
    ]

    raw_documents, n = get_books(books_to_download)

    documents = [preprocess(document, n, i) for i, document in enumerate(raw_documents, 1)]

    tf_document = calculate_tf(documents, n)

    tf_idf_document = calculate_tf_idf(tf_document, n)

    vectorized_document = vectorize_tf_idf(tf_idf_document, n)

    cos_sim = cosine_comparison(vectorized_document[0], vectorized_document)

    print(cos_sim[1])

    # Example 2, compare one book to a collection and show which are most similar

    book_to_compare = 'https://www.gutenberg.org/files/7766/7766-h/7766-h.htm'

    raw_documents, n, titles = get_all_books(book_to_compare)
    documents = [preprocess(document, n, i) for i, document in enumerate(raw_documents, 1)]
    tf_document = calculate_tf(documents, n, show=False)
    tf_idf_document = calculate_tf_idf(tf_document, n, show=False)
    vectorized_document = vectorize_tf_idf(tf_idf_document, n)
    cos_sim = cosine_comparison(vectorized_document[-1], vectorized_document)

    print('From our major database of books, in descending order, the most similar are:')
    results = sorted(zip(titles, cos_sim), key=lambda x: x[1], reverse=True)[1:]
    for result in results:
        print(f'{result[0]}: Score {round(result[1], 4)}')


if __name__ == "__main__":
    main()
