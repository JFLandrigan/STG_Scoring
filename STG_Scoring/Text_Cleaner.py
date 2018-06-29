#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Import string package for removal of punctuation
import string
#Import the spellchecker function for spelling correction
from enchant.checker import SpellChecker


class Cleaner():
    '''
    Cleaner class: class to perform preprocessing cleaning of the reflections
    top leavel function to be called is Get_Clean_Tokens
    '''
    
    #initialize the class
    def _init_(self):
        pass
    
    
    def _strip_punctuation(self, s):
        '''
        strip_punctuation: function to remove punction from the reflection strings
        Expects: string Cleaned_Reflection
        Returns: returns string without punctuation
        '''
        return([x.strip(string.punctuation) for x in s.split()])
    
    
    def _remove_blank(self, x):
        '''
        remove_blank: funciton to remove blank strings from token sets
        Expects: list containing word tokens (Word_Tokenized_Reflection)
        Returns: tokenized list with blank spaces removed
        '''
        return(list(filter(None, x)))
    
    #set_lower: function to put all strings in lower case
        #Expects: String
        #Returns string in lower case
    def _set_lower(self, x):
        '''
        set_lower: function to set strings to lower case
        Expects: string 
        Returns: string in all lower case
        '''
        return(x.lower())
    
    #string_cleaner: function for cleaning the reflections
        #Expects a dataframe containg the reflections (column name: reflection)
        #retruns a dataframe with the cleaned strings
    def spell_check(self, df):
        '''
        spell_check: function for correcting the spelling of the reflections
        Expects: pandas dataframe with column titled Reflection
        Returns: the dataframe with column Corrected_Reflection (spelling corrected reflection)
        '''
        
        Corr_RF = []
        
        #Grab each individual reflection
        for refl in df["reflection"]:
            #Check to see if the words are in the dictionary
            chkr = SpellChecker("en_US", refl)
            #for the identified errors or words not in dictionary get the suggested correction
            #and replace it in the reflection string
            for err in chkr:
                if len(err.suggest()) > 0:
                    sug = err.suggest()[0]
                    err.replace(sug)
            Corr_RF.append(chkr.get_text())
        df["Corrected_Reflection"] = Corr_RF
        #return the dataframe with the new corrected reflection column
        return(df)
        
    
    def Get_Clean_Tokens(self, df):
        '''
        Get_Clean_Tokens: top level funciton for cleaning and preprocessing in Text_Cleaner
        Expects: a pandas dataframe with column titled Reflection containing strings
        Returns: the datframe with the cleaned reflection string and tokenized lists
        '''
        
        #perform the spell checking
        df = self.spell_check(df)
        #set the strings to all lower
        df["Corrected_Reflection"] = df["Corrected_Reflection"].apply(self._set_lower)
        #Tokenize the reflections (also removes punctuation and empty strings)
        df["Word_Tokenized_Reflection"] = df["Corrected_Reflection"].apply(self._strip_punctuation)
        df["Word_Tokenized_Reflection"] = df["Word_Tokenized_Reflection"].apply(self._remove_blank)
    
        return(df)