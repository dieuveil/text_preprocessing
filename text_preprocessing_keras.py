# TEXT PREPROCCESSING WITH KERAS

# Import libraries
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import hashing_trick

# Creating text data
def get_data():
    text = 'This is all I have as data to train my deep learning models and reach my purpose.'
    return text

def get_data2():
    text = ["This is the first text from our corpus that we are going to vectorize.",
             "The second text that we want to vectorize is this one and the third one is coming.",
             "This is the third and the last one. Let do it and get the job done"] 
    return text

# Tokenizing text data
def tokenization():
    text = get_data()
    token = text_to_word_sequence(text)
    return token

# One_hot encoding
def vectorize_with_one_hot():
    token = tokenization()
    text = get_data()
    word = set(token)
    vocab_size = len(word)
    print(vocab_size)
    result = one_hot(text, round(vocab_size*1.3))
    print(result)

# Hash encoding
def vectorize_with_hash():
    token = tokenization()
    text = get_data()
    word = set(token)
    vocab_size = len(word)
    print(vocab_size)
    result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
    print(result)

def executor():
    vectorize_with_one_hot()
    vectorize_with_hash()



# USING tokenizer API
from keras.preprocessing.text import Tokenizer

def vectorize_with_tokenizer():
    docs = get_data2()
    t = Tokenizer()
    t.fit_on_texts(docs)
    print('{0}\n {1}\n {2}\n {3}'.format(t.word_counts, t.word_docs, t.document_count, t.word_index))
    encoded_docs = t.texts_to_matrix(docs, mode='count')
    print(encoded_docs)

vectorize_with_tokenizer()
