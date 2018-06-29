#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import pandas for working with dataframes
import pandas as pd
#Import numpy for generating numpy arrays to be added onto pandas dataframe
import numpy as np
#Import gensim for performing word2vec calculations
import gensim
#Imports the nltk stopword corpus 
from nltk.corpus import stopwords
#Import counter for creating dict of pos tag frequencies
from collections import Counter
#Import nltk for POS tagging
import nltk
#Import pickle for loading pickled files 
import pickle


def POS_Counters(df):
    '''
    POS_Counters: function for collecting counts of parts of speech
    Expects: dataframe containing column Word_Tokenized_Reflection
    Returns: the dataframe with the POS count columns appended
    '''

    #Generate the POS tags
    df["POS_Tags"] = df["Word_Tokenized_Reflection"].apply(nltk.pos_tag)
    #get counts for each of the tags
    df["POS_Counts"] = df["POS_Tags"].apply(lambda x: Counter(tag for word,tag in x))
   
    #generate the individual counts for the reflections
    #common-carrier cabbage knuckle-duster Casino afghan
    df["Num_Nouns"] = df["POS_Counts"].apply(lambda x: x['NN'])
    #hers herself him himself hisself it itself me myself
    df["Num_ProNounPers"] = df["POS_Counts"].apply(lambda x: x['PRP'])
    #her his mine my our
    df["Num_ProNounPosses"] = df["POS_Counts"].apply(lambda x: x['PRP$'])
    #verb present participle telegraphing stirring focusing angering 
    df["Num_VerbING"] = df["POS_Counts"].apply(lambda x: x['VBG'])
    #dipped pleaded swiped regummed
    df["Num_PastVerbs"] = df["POS_Counts"].apply(lambda x: x['VBD'])
    #bleaker braver breezier briefer brighter
    df["Num_AdjComp"] = df["POS_Counts"].apply(lambda x: x['JJR'])
    #further gloomier grander graver
    df["Num_AdvComp"] = df["POS_Counts"].apply(lambda x: x['RBR'])
    #modal auxiliary can cannot could couldn't dare may might must need ought shall should shouldn't will would
    df["Num_Modal"] = df["POS_Counts"].apply(lambda x: x['MD'])

    return(df)
    
    
def bigram_counter(df):
    '''
    bigram_counter: function to count the number of key trigrams in each reflection
    Expects: dataframe containing column Cleaned_Reflection_String
    Returns: dataframe with bigram_count column containing the number of trigrams appearing in a reflection
    '''
    
    #load the bigrams
    bis = pickle.load(open('data_files/bigrams.pickle', 'rb'))
    
    #get the counts of the occurences of the bigrams
    bi_counts = []
    for refl in df["Cleaned_Reflection_String"]:
        numBi = np.array([])
        refl = refl.lower()
        for bi in bis:
            numBi = np.append(numBi, refl.count(bi))
    
        bi_counts.append(numBi.sum())
    

    df["Bigram_Count"] = bi_counts
    
    return(df)

    
def trigram_counter(df):
    '''
    trigram_counter: function to count the number of key trigrams in each reflection
    Expects: dataframe containing column Cleaned_Reflection_String
    Returns: dataframe with trigram_count column containing the number of trigrams appearing in a reflection
    '''
    
    #load the trigrams
    tris = pickle.load(open('data_files/trigrams.pickle', 'rb'))
    
    #get the counts of the occurences of the trigrams
    tri_counts = []
    for refl in df["Cleaned_Reflection_String"]:
        numTri = np.array([])
        refl = refl.lower()
        for tr in tris:
            numTri = np.append(numTri, refl.count(tr))
    
        tri_counts.append(numTri.sum())

    df["Trigram_Count"] = tri_counts

    return(df)

    

def startegy_word_count(df):
    '''
    strategy_word_count: function to get the total number of task words appearing the reflection
    Expects: a pandas dataframe with a column entitled 'Cleaned_Reflection_String
    Returns: the dataframe with a column appended to it containing strategy word counts
    '''
    
    #read in the dataframes containing the list of strategy and task words
    strat_wrds = pd.read_csv("data_files/Strategy_Words.csv")
    
    strat_wrd_counts = []
    for refl in df['Cleaned_Reflection_String']:
        numSWrd = np.array([])
        refl = refl.lower()
        for wrd in strat_wrds['Strategy_Words']:
            numSWrd = np.append(numSWrd, refl.count(wrd))
        
        strat_wrd_counts.append(numSWrd.sum())
    
    df["Strategy_Word_Count"] = strat_wrd_counts
    
    return(df)
    
    
def task_word_count(df):
    '''
    task_word_count: function to get the total number of task words appearing the reflection
    Expects: a pandas dataframe with a column entitled 'Cleaned_Reflection_String
    Returns: the dataframe with a column appended to it containing task word counts
    '''
    
    #load in the task words
    task_wrds = pd.read_csv("data_files/Task_Words.csv")
    
    #loop the reflections and count the occurences of each task word
    task_wrd_counts = []
    for refl in df['Cleaned_Reflection_String']:
        numTWrd = np.array([])
        refl = refl.lower()
        for wrd in task_wrds['Task_Words']:
            numTWrd = np.append(numTWrd, refl.count(wrd))
        
        #append the sum total of task words that appeared 
        task_wrd_counts.append(numTWrd.sum())
    
    df["Task_Word_Count"] = task_wrd_counts
    
    return(df)
    

def word_vec(df):
    '''
    word_vec: function to calculate the word2vec summation and avg scores 
    Expects: dataframe containing column Word_Tokenized_Reflection
    Returns: the dataframe with the sum, avg, min and max word2vec columns
    '''
    
    #remove stop words word2vec having issues with 'a'
    stop_words = set(stopwords.words('english'))
    df['Token_Set'] = df['Word_Tokenized_Reflection'].apply(lambda x: [item for item in x if item not in stop_words])
    
    #load word2vec model
    floc = 'data_files/'
    fname = 'GoogleNews-vectors-negative300.bin'
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(floc + fname, binary = True)

    #initialize a dictionary to contain the values for each reflection
    refl_w2v_fts = {"sums": [], "avgs": [], "mins": [], "maxs": []}
    #Loop through the reflections and get the vector representation values
    for refl in df["Token_Set"]:
        tmp_ft_dict = {"sums": np.array([]), "avgs": np.array([]), "mins": np.array([]), "maxs": np.array([])}

        #for each word in the reflection get the sum of the word2vec rep
        for wrd in refl:
            #try to get sum of the wrd matrix from word2vec
            try:
                #get the word embedding matrix
                word_vector = word2vec[wrd]
                
                tmp_ft_dict["sums"] = np.append(tmp_ft_dict["sums"], sum(word_vector))
                tmp_ft_dict["avgs"] = np.append(tmp_ft_dict["avgs"], sum(word_vector)/len(word_vector))
                tmp_ft_dict["mins"] = np.append(tmp_ft_dict["mins"], min(word_vector))
                tmp_ft_dict["maxs"] = np.append(tmp_ft_dict["maxs"], max(word_vector))
                
            #if the word wasn't in the vocabulary just move on to the next one
            except Exception as e:
                continue
            
        #append the values to the lists containing all the values
        refl_w2v_fts["sums"].append(tmp_ft_dict["sums"].sum())
        refl_w2v_fts["avgs"].append(tmp_ft_dict["avgs"].mean())

        if len(tmp_ft_dict["mins"]) > 0:
            refl_w2v_fts["mins"].append(tmp_ft_dict["mins"].min())
            refl_w2v_fts["maxs"].append(tmp_ft_dict["maxs"].max())
        else:
            refl_w2v_fts["mins"].append(float('nan'))
            refl_w2v_fts["maxs"].append(float('nan'))
    
    #Append each list to the dataframe as a column
    #Since there are means of 0 in the column replace these with 0
    df["W2V_Sum"] = refl_w2v_fts["sums"]
    df["W2V_Avg"] = refl_w2v_fts["avgs"]
    df["W2V_Avg"].fillna(value = 0, inplace = True)
    df["W2V_Min"] = refl_w2v_fts["mins"]
    df["W2V_Min"].fillna(value = 0, inplace = True)
    df["W2V_Max"] = refl_w2v_fts["maxs"]
    df["W2V_Max"].fillna(value = 0, inplace = True)
                
    return(df)
    

def feature_generation(df):
    '''
    feature_generation: top level function to be called in module
    Expects: a pandas dataframe with the columns appended to it by Text_Cleaner module
    Returns: the dataframe with the feature columns appened to it
    '''
    
    #Get the number of words in the reflection 
    df["Num_Words"] = df["Word_Tokenized_Reflection"].apply(len)
    #Get the columns containing the numbers of POS
    df = POS_Counters(df)
    
    #take the celaned word_tokenized reflections and join them down to single string again
        #for bi and tri 
    df["Cleaned_Reflection_String"] = df["Word_Tokenized_Reflection"].apply(" ".join)
    #Get the number of key bigrams in the reflection
    df = bigram_counter(df)
    #Get the number of key trigrams in the reflection
    df = trigram_counter(df)
    
    #Get the counts of ket strategy and task words in the reflection
    df = startegy_word_count(df)
    df = task_word_count(df)
    
    #Get the word2vec features
    df = word_vec(df)
    
    return(df)
    
