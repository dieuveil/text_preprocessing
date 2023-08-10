# PREPROCESSING TEXT WITH SPACY
# Importance of spacy : natural language understanding, information extraction, ----
# ---- preprocess text for deep learning 
# It's a free library for advanced NLP
# Build application that process and understand large volume of text

# Import libraries
import spacy

# Get data
def get_data():
    nlp = spacy.load("en_core_web_sm")
    doc = nlp("Apple is looking at buying U.K. startup for $1 billion")
    return doc

# Linguistic Features
def linguistic_features():
    doc = get_data()
    for token in doc:
        print('{0} - {1} - {2} - {3} - {4} - {5}'.format(token.text, token.tag_, 
                                                         token.lemma_, token.pos_, 
                                                         token.shape_, token.is_stop,
                                                         ))

# Entity recognition
def entity_recognition():
    doc = get_data()
    for entity in doc.ents:
        print(entity.text, entity.label_)
        

# Executing functions
def executor():
    entity_recognition()
    linguistic_features()

executor()
