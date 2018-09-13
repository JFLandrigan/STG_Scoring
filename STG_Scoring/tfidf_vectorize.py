#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#inport the tfidf vectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
#import string 
import string
#import the stopword dictionary from nltk
from nltk.corpus import stopwords
#import pickle for saving the vectorizer
import pickle


def text_process(mess):
    '''
    text_process: function to perform preprocessing specific to the TfidfVectorizer
    Expects: string from Cleaned_Reflection
    Returns: tokenized list of words without stop words
    '''
    
    #split up the string
    nopunc = [x.strip(string.punctuation) for x in mess.split()]
    #remove the stop words
    nonstop = [word for word in nopunc if word.lower() not in stopwords.words('english')]
    # remove blank spaces
    return(list(filter(None, nonstop)))
    

def frequency_vectorize(df):
    '''
    frequency_vectorize: function to generate tfidf sparse matrix and save out vectorizer
        function is used during the TRAINING process
    Expects: pandas dataframe containing Corrected_Reflection column
    Returns: tfidf sparse matrix 
    '''
    vectorizer = TfidfVectorizer(analyzer = text_process)
    #initialize the vectorizer
    vectorizer.fit(df["Corrected_Reflection"])
    
    #save the vectorizor out
    #save out the feature generated randomforest
    modSave = open("data_files/tfidf_vectorizer.pickle", "wb")
    pickle.dump(vectorizer, modSave)
    modSave.close()
    
    #get the feature matrix for the tfidf and return it
    return(vectorizer.transform(df["Corrected_Reflection"]))


def get_tfidf(df):
    '''
    get_tfidf: function to generate the tfidf sparse matrix when generating new scores
    Expects: pandas datframe contained Corrected_Reflection column
    Returns: tfidf sparse matrix
    '''
    #load the vectorizer
    vectorizer = pickle.load(open("data_files/tfidf_vectorizer.pickle", 'rb'))
    #return the sparse matrix
    return(vectorizer.transform(df["Corrected_Reflection"]))
