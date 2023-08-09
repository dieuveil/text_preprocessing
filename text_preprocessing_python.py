# TEXT PREPROCESSING WITH PYTHON

# Libraries importation
import string
import re


# Load text data
def get_data():
    filename = 'training.txt'
    file = open(filename, 'rt')
    text = file.read()
    file.close()
    return text

# Split text into words by white space
def text_split():
    text = get_data()
    words = text.split()
    return words

# Remove punctuation
def remove_punctuation():
    words = text_split()
    #print(string.punctuation)
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    stripped = [re_punc.sub('', w) for w in words]
    return stripped

# Normalize or standardize case
def normalize_case():
    stripped = remove_punctuation()
    normal_data = [s.lower() for s in stripped]
    print(normal_data)

normalize_case()