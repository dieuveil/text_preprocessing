# TEXT PREPROCESSING WITH SCIKIT-LEARN

 # Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer


# Selft created data
def get_data():
    # Collection of documents
    text = ["This is the first text from our corpus that we are going to vectorize.",
             "The second text that we want to vectorize is this one and the third one is coming.",
             "This is the third and the last one. Let do it and get the job done"]
    text2 = ["This is the first text from our corpus that we are going to vectorize."]
    return text

# Vectorizing with the help of CountVectorizer
def corpus_vectorizer_with_countvectorizer():
    text = get_data()
    vectorizer = CountVectorizer()
    vectorizer.fit(text)
    vector = vectorizer.transform(text)
    print(vector.toarray())

# Vectorizing with the help of TfidfTransformer
def corpus_vectorizer_with_tfidftransformer():
    text = get_data()
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)
    vector = vectorizer.transform(text)
    print(vector.toarray())

# Vectorizing with the help of HashingVectorizer
def corpus_vectorizer_with_hashingvectorizer():
    text = get_data()
    vectorizer = HashingVectorizer(n_features=30)
    vectorizer.fit(text)
    vector = vectorizer.transform(text)
    print(vector.toarray())


# Executing these functions
def executor():
    corpus_vectorizer_with_countvectorizer()
    corpus_vectorizer_with_tfidftransformer()
    corpus_vectorizer_with_hashingvectorizer()

executor()