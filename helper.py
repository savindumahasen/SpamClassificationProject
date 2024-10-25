import pandas as pd
import numpy as np
import re
import string
import pickle

from nltk.stem import PorterStemmer

ps = PorterStemmer()

## Load the model
with open ('spam_classifier.pkl', 'rb') as f:
    model1 = pickle.load(f)


## Load the stop words
with open ('static/model/corpora/stopwords/english', 'r') as file:
    sw=file.read().splitlines()

## Load the vocabulary
with open ('static/model/vocabulary.txt', 'r') as vocabulary_file:
    vocabulary_text = vocabulary_file.read().splitlines()

def remove_punctuations(text):
    for punctuation in string.punctuation:
        text =text.replace(punctuation, '')
    return text

def preprocessing(text):
    data = pd.DataFrame([text], columns=['sms'])
    data['sms']=data['sms'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['sms']=data['sms'].apply(lambda x: " ".join(re.sub(r'^https?:\/\/.*[\r\n]*','', x,flags=re.MULTILINE) for x in x.split()))
    data["sms"]=data["sms"].str.replace('/d+','', regex=True)
    data["sms"]=data["sms"].str.replace('/d+','', regex=True)
    data["sms"]= data["sms"].apply(lambda x: " ".join(x for x in x.split() if x not in sw))
    data["sms"]=data["sms"].apply(lambda x: " ".join(ps.stem(x) for x in x.split()))
    return data["sms"]


def vectorizer(ds):
    vectorizer_list = []
    vocab_set = set(vocabulary_text)  # Use a set for faster lookup
    
    for sentence in ds:
        sentence_list = np.zeros(len(vocabulary_text), dtype=np.float16)  # Use np.float16 to reduce memory usage
        sentence_words = set(sentence.split())  # Split and convert sentence to a set to speed up membership checking
        
        for i, word in enumerate(vocabulary_text):
            if word in sentence_words:  # Check if word exists in the sentence
                sentence_list[i] = 1
                
        vectorizer_list.append(sentence_list)
    
    # Convert the list to a numpy array only once
    vectorizer_list_new = np.asarray(vectorizer_list, dtype=np.float16)
    
    return vectorizer_list_new

def get_prediction(vectorizer_text):
    predictions = model1.predict(vectorizer_text)
    if predictions[0] ==1:
        return "Spam"
    else:
        return "Hamp"
