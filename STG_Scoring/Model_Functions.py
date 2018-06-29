#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import pickle for loading and saving out models
import pickle
import pandas as pd
import Text_Cleaner
import Feature_Generator
import tfidf_vectorize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


def load_model(flname):
    '''
    load_model function to load pickled models
    Expects: string containing the file name and path is necessary
    Returns: the model
    '''
    return(pickle.load(open(flname, 'rb')))
    
    
def predict_scores(ftgen_df, tfidf_mat):
    '''
    predict_scores: function to score the new reflections
    Expects: the dataframe containing the engineered functions
    Returns: a list containing the rounded scores 
    '''
    
    #Load the model for scoring the reflections
    ftgen_mod = load_model("data_files/ftgen_mod.pickle")
    tfidf_mod = load_model("data_files/tfidf_mod.pickle")
    lin_mod   = load_model("data_files/lin_mod.pickle")
    
    
    #define the predictor columns
    predictors = ["Num_Words", "Num_Nouns", "Num_ProNounPers", "Num_ProNounPosses",
              "Num_VerbING", "Num_PastVerbs", "Num_Modal", "Num_AdvComp", "Num_AdjComp", "Trigram_Count", 
              "Bigram_Count", "Strategy_Word_Count", "Task_Word_Count", "W2V_Sum", "W2V_Avg", "W2V_Min", "W2V_Max"]

    #get the predictions from the feature generated model and the tfidf model
    gen_pred = ftgen_mod.predict(ftgen_df.loc[:,predictors])
    tfidf_pred = tfidf_mod.predict(tfidf_mat)
    #combine the predictions from the models into a single dataframe
    #combine the predictions 
    d = {"gen": gen_pred, "tfidf" : tfidf_pred}
    comboPreds = pd.DataFrame(d)
    #get the final predictions from the linear model
    new_scores = lin_mod.predict(comboPreds)
    
    #since the model is a regression it rounds the values before writing them back to sql
    new_scores = new_scores.round()

    #if a score is greater then 4 or less then 1 post rounding then set to 1 or 4
    new_scores[new_scores < 1] = 1
    new_scores[new_scores > 4] = 4
    
    #return the scores
    return(new_scores)
    
    

def train_model(data_fl):
    '''
    train_model: function to train the models and save out the model states
    Expects: the file name of the data train the model on
    DOES NOT RETURN anything only saves out the model states
    '''
    
    #read in the data
    df = pd.read_csv(data_fl, encoding = "ISO-8859-1")
    
    #Aggregate the data by mean score to make sure there are no duplicate reflections
    #when training
    df["reflection"] = df["reflection"].fillna("NR")
    #replace the blank spaces those they don't get dropped
    df["Prompt"] = df["Prompt"].fillna("XXX")
    #add on the reset_index so that columns don't become index values
    df = df.groupby(["Id","Prompt", "reflection"]).mean().reset_index()
    #replace the "NR" with blank so when tokenize they don't end up as words
    df["reflection"] = df["reflection"].replace(to_replace = "NR", value = "")
    #drop the reviewer id from the dataframe
    df = df.drop(["Reviewer ID"], axis = 1)
        
    #clean the data and get the associated features
    #first create object of the class 
    cleaner = Text_Cleaner.Cleaner()
    df = cleaner.Get_Clean_Tokens(df)
   
    #Get the features
    df = Feature_Generator.feature_generation(df)
    
    #get the tfidf vectorized representation
    ref_vec = tfidf_vectorize.frequency_vectorize(df)
    
    #define the predictor columns
    predictors = ["Num_Words", "Num_Nouns", "Num_ProNounPers", "Num_ProNounPosses",
              "Num_VerbING", "Num_PastVerbs", "Num_Modal", "Num_AdvComp", "Num_AdjComp", "Trigram_Count", 
              "Bigram_Count", "Strategy_Word_Count", "Task_Word_Count", "W2V_Sum", "W2V_Avg", "W2V_Min", "W2V_Max"]
    
    combo_df = pd.concat([df.loc[:,predictors], pd.DataFrame(ref_vec.toarray())], axis = 1)
    combo_df["score"] = df["score"]
    combo_df = combo_df[combo_df["Num_Words"] > 0]
    
    #reassign the tfidf to new dataframe
    dropCols = [p for p in predictors]
    dropCols.append("score")
    tfidf_df = combo_df.drop(dropCols, axis = 1)
    
    
    #training pipeline 
    #ft gen 
    rfMod_gen = RandomForestRegressor(criterion = "mae", min_samples_split=3)
    rfMod_gen.fit(combo_df.loc[:,predictors], combo_df["score"])
    y_predicted_gen = rfMod_gen.predict(combo_df.loc[:,predictors])
    #tfidf pred
    rfMod_tfidf = RandomForestRegressor(criterion = "mae", min_samples_split=4)
    rfMod_tfidf.fit(tfidf_df, combo_df["score"])
    y_predicted_tfidf = rfMod_tfidf.predict(tfidf_df)
    #combine the predictions 
    d = {"gen": y_predicted_gen, "tfidf" : y_predicted_tfidf}
    comboPreds = pd.DataFrame(d)
    #train the linear mod
    linMod = LinearRegression()
    linMod.fit(comboPreds, combo_df["score"])

    #save out the feature generated randomforest
    modSave = open("data_files/ftgen_mod.pickle", "wb")
    pickle.dump(rfMod_gen, modSave)
    modSave.close()
    #save out the tfidf randomforest
    modSave = open("data_files/tfidf_mod.pickle", "wb")
    pickle.dump(rfMod_tfidf, modSave)
    modSave.close()
    #save out the linear model
    modSave = open("data_files/lin_mod.pickle", "wb")
    pickle.dump(linMod, modSave)
    modSave.close()
