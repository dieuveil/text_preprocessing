# TEXT PREPROCESSING WITH NLTK

# Libraries importation
import string
import re
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


# Load text data
def get_data():
    filename = 'training.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text

# Split text into sentences
def split_into_sentences():
    text = get_data()
    sentences = sent_tokenize(text)

# Split text into words
def split_into_words():
    text = get_data()
    words = word_tokenize(text)
    return words

# Remove punctions
def remove_punctions():
    words = split_into_words()
    stripped = [word for word in words if word.isalpha()]
    return stripped

# Normalize case
def text_normalization():
    stripped = remove_punctions()
    normal_words = [st.lower() for st in stripped]
    return normal_words

# Remove stop words
def remove_stop_words():
    normal_words = text_normalization()
    stop_words = set(stopwords.words('english'))
    final_token = [nw for nw in normal_words if not nw in stop_words]
    return final_token

# Stemming : Reducing words to their root or base (fishing, fished, fisher -> fish)
def words_stemming():
    final_token = remove_stop_words()
    porter = PorterStemmer()
    stemmed = [porter.stem(word) for word in final_token]
    print(stemmed)



words_stemming()

