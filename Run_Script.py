#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import Text_Cleaner
import Feature_Generator
import Model_Functions
import tfidf_vectorize

import pandas as pd
import numpy as np
import psycopg2
import psycopg2.extras

#If wanted update the model based on new data else load the existing model
modUpdate = input("Would you like to update the model? [y/n] ")
if modUpdate == "y":
    Model_Functions.train_model("data_files/Reflection Audit Data - Reflections.csv")

#Read in the data that does not have a score
# Connect to make queries using psycopg2
dbname = ''
username = '' 
con = None
con = psycopg2.connect(database = dbname, user = username)

# query:
sql_query = "SELECT * FROM reflections WHERE Score IS NULL;"
refl_df = pd.read_sql_query(sql_query,con)

#Clean the data and get the token sets from the reflection
cleaner = Text_Cleaner.Cleaner()
refl_df = cleaner.Get_Clean_Tokens(refl_df)

#Get the features used for scoring the model
refl_df = Feature_Generator.feature_generation(refl_df)
#get the tfidf matrix
tfidf_mat = tfidf_vectorize.get_tfidf(refl_df)
#Get the scores for the reflections
new_scores = Model_Functions.predict_scores(refl_df, tfidf_mat)

#Put the data to be updated in the table in a list of tuples includes:
# score
# reviewer id: 999 for automated score
# reflection_id 
tpDat = zip(new_scores.tolist(),  np.repeat(999, 
            refl_df.shape[0]).tolist(), 
            refl_df["reflection_id"].tolist())

#Write the results back out to the database
#sql_query = """UPDATE refelctions SET Score = %s WHERE reflection_id = %s"""

sql_query = """UPDATE reflections SET Score = data.score, reviewer_id = data.reviewer_id  
FROM (VALUES %s) AS data (score, reviewer_id, reflection_id) 
WHERE reflections.reflection_id = data.reflection_id"""
# create a new cursor
cur = con.cursor()
psycopg2.extras.execute_values(cur, sql_query, tpDat)
# Commit the changes to the database
con.commit()
# Close communication with the PostgreSQL database
cur.close()