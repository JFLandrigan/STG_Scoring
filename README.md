# STG_Scoring

This package contains files and funcitons specific for the scoring of the quality of reflections from users of the Sown To Grow platform (https://www.sowntogrow.com/).

Note that this repo only contains the code for the scoring pipeline but does not contain the data files necessary for generating features or initializing the models as that data is proprietary.

Included in the package:

    - Text_Cleaner.py: module for preprocessing
    
    - Feature_Generator.py: module for generating features from the reflections
    
    - Model_Functions.py: module containing functions for training models and generating new scores
    
    - tfidf_vectorize.py: module for performing functions specific to generating tf-idf matrices
    
    - Run_Script.py: example script for using the modules to score reflections. 

Contributors:
Jon-Frederick Landrigan 
