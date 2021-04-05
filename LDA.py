#!/usr/bin/env python

''' ---------------- About the script ----------------

Assignment 5: LDA topic modeling: exploration of topics in American television sitcom 'Friends' (10 seasons and 236 episodes)

This script cleans and preprocess the full script of Friends series and investigates the topics of the show with the help
of LDA model. It outputs pyLDAvis topic graphs as html files for the full script and each character separately. Also, 
it saves .csv files for full script and each character to further investigate the topics and their keywords.

positional arguments: 
    -f,   --input_file      directory of the input file (txt files)

    
Example:    
    $ python LDA.py -f data/FRIENDS_TV_script

    

'''




"""---------------- Importing libraries ----------------
"""

# standard library
import sys,os
sys.path.append(os.getcwd())
sys.path.append(os.path.join(".."))
from pprint import pprint
from pathlib import Path
import io
import glob

# data and nlp
import pandas as pd
import spacy
nlp = spacy.load("en_core_web_sm", disable=["ner"])

# visualisation
import pyLDAvis.gensim
#pyLDAvis.enable_notebook()
import seaborn as sns
from matplotlib import rcParams
# figure size in inches
rcParams['figure.figsize'] = 20,10


# LDA tools
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from utils import lda_utils

# warnings
import logging, warnings
warnings.filterwarnings('ignore')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import argparse

import re





def sepExp(text):
    '''
    separates staging directions from conversations and returns
    only conversations
    
    '''
    regex = re.compile(".*?\((.*?)\)")
    expression = re.findall(regex, text)
    expression = ','.join(expression)
    diag_filt = re.sub("[\(\[].*?[\)\]]", "", text)
    return diag_filt
    
    
    
    
def filter_words(string_list):
    '''
    filters out frequent but semantically
    meaningless words
    
    '''   
    words= ['guy', 'chandler:', 'monica:', 'ross:', 'joey:', 'phoebe:', 'rachel:', 'guys?', 'guys?!', 'guys,', "guy's", "guys'd", 'guy!"', 'guys', "guy'", "guys'", 'guys...',  'guys.', 'good', 'better', 'best', 'guys!', 'guy.', 'guy!', 'guy?', 'guy?"', 'guy..?', 'guy,', 'thing', 'things', 'thing.', 'things.', 'thing!']
    filtered_out = [" ".join([w for w in t.split() if not w in words]) for t in string_list]
    return filtered_out
    
    
def LDA(chunks, csv_name, vis_name):   
    '''
    data preprocessing, lda model, graphs and csv files
       
    '''
        
    # Build the bigram and trigram models
    bigram = gensim.models.Phrases(chunks, min_count=10, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[chunks], threshold=100) 

    # fitting the models to the data
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    data_processed = lda_utils.process_words(chunks,nlp, bigram_mod, trigram_mod, allowed_postags=["NOUN"])
    
    
    # Create Dictionary
    id2word = corpora.Dictionary(data_processed)

    # Create Corpus: Term Document Frequency
    corpus = [id2word.doc2bow(text) for text in data_processed]
    
    
    # Build LDA model
    lda_model = gensim.models.LdaMulticore(corpus=corpus,           # vectorised corpus - list of lists of tuples
                                           id2word=id2word,         # gensim dictionary - mapping words to IDS
                                           num_topics=15,            # number of topics
                                           random_state=100,        # set for reproducability
                                           chunksize=10,            # batch data for efficiency
                                           passes=10,               # number of full passes over data
                                           iterations=100,          # related to document rather than corpus
                                           per_word_topics=True,    # define word distributions 
                                           minimum_probability=0.0) # minimum value


    
    # Compute Perplexity
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

    # Compute Coherence Score
    coherence_model_lda = CoherenceModel(model=lda_model, 
                                         texts=data_processed, 
                                         dictionary=id2word, 
                                         coherence='c_v')

    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)

    
    
 
    df_topic_keywords = lda_utils.format_topics_sentences(ldamodel=lda_model, 
                                                          corpus=corpus, 
                                                          texts=data_processed)

    # Format
    df_dominant_topic = df_topic_keywords.reset_index()
    df_dominant_topic.columns = ['Chunk_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
    
    # Display setting to show more characters in column
    pd.options.display.max_colwidth = 100
    sent_topics_sorteddf = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_keywords.groupby('Dominant_Topic')

    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf = pd.concat([sent_topics_sorteddf, 
                                          grp.sort_values(['Perc_Contribution'], ascending=False).head(1)], 
                                          axis=0)

    # Reset Index    
    sent_topics_sorteddf.reset_index(drop=True, inplace=True)
    # Format
    sent_topics_sorteddf.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Representative Text"]
    
    #defining .csv file name
    out_file_name = csv_name
    # creating an output folder to save networks.csv file, if it does not already exist 
    if not os.path.exists("output"):
            os.makedir("output")
    # defining full filepath to save networks.csv file 
    outfile = os.path.join("output", out_file_name)
    # saving a dataframe as .csv
    sent_topics_sorteddf.to_csv(outfile)
    
    # Show
    sent_topics_sorteddf.head(10)
    
    
    # visualize topics
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.id2word)
    
   
    # save as html
    pyLDAvis.save_html(vis, vis_name)

    
    
    
    

"""---------------- Main script ----------------
"""


def main():
    
    """------ Argparse parameters ------
    """
    
    # instantiating the ArgumentParser  object as parser 
    parser = argparse.ArgumentParser()
    # adding positional arguments
    parser.add_argument("-f" , "--input_file", help="input file directory")
    #parser.add_argument("-n" , "--csv_name", help="full csv file name")
    # adding optional arguments
    #parser.add_argument("-w", "--weight", help="cut-off point to filter data based on a certain edge weight", type=int)
      
    # parsing the arguments
    args = parser.parse_args()
        
   


    """------ Loading data and initial preprocessing ------
    """
    
    # texts directory
    path = os.path.join(args.input_file)
    # empty list to store text files
    script = [] 
    # read text files 
    for text in Path(path).glob("*.txt"):
        with open(text) as t:
            content = t.read()
            # append to a list
            script.append(content)
    
        
    # convert a list of all the scripts to a list of strings and then join its elements (it is needed for aplying the function below above)
    stringList = ' '.join([text for text in script ])
    
    # data cleaning: separating staging direction from conversation
    stringList = sepExp(stringList)
    
    # converting to lowercase so I can filter strings based on the character, if needed
    stringList = stringList.lower()
    
    # split on new lines
    stringList = stringList.split('\n')
    
    # filter our words without much semantic meaning
    string_list = filter_words(stringList)
    
    
    
    """------ LDA model on the full script ------
    """
        
    print("[INFO] running full script model ...")  
    
    # create chunks of 40 sentences (because sentences are quite short) at a time
    chunks = []
    for index in range(0, len(string_list), 40):
        chunks.append(' '.join(string_list[index:index+40]))
        
    LDA(chunks, "topics_full.csv", "full_vis.html")
    
    
    

    
    """------ Filtering based on character and running separate models for each ------
    """
    
    print("[INFO] running Chandler´s model ...")
    
    # filtering only Chandler´s lines
    filtered = [x for x in stringList if x.startswith('chandler:')]
    # filter our words without much semantic meaning
    filtered_out = filter_words(filtered)
    # create chunks of 30 sentences at a time
    chunks_filtered = []
    for index in range(0, len(filtered_out), 30):
        chunks_filtered.append(' '.join(filtered_out[index:index+30]))
        
    LDA(chunks_filtered, "topics_chandler.csv", "chandler_vis.html")
    
    
    
    print("[INFO] running Monica´s model ...")
    
    # filtering only Monica´s lines
    filtered = [x for x in stringList if x.startswith('monica:')]
    # filter our words without much semantic meaning
    filtered_out = filter_words(filtered)
    # create chunks of 30 sentences at a time
    chunks_filtered = []
    for index in range(0, len(filtered_out), 30):
        chunks_filtered.append(' '.join(filtered_out[index:index+30]))
               
    LDA(chunks_filtered, "topics_monica.csv", "monica_vis.html")
    
    
    
    print("[INFO] running Ross´s model ...")

    # filtering only Ross´s lines
    filtered = [x for x in stringList if x.startswith('ross:')]
    # filter our words without much semantic meaning
    filtered_out = filter_words(filtered)
    # create chunks of 30 sentences at a time
    chunks_filtered = []
    for index in range(0, len(filtered_out), 30):
        chunks_filtered.append(' '.join(filtered_out[index:index+30]))
               
    LDA(chunks_filtered, "topics_ross.csv", "ross_vis.html")
    
    
    
    
    print("[INFO] running Rachel´s model ...")
    
    # filtering only Rachel´s lines
    filtered = [x for x in stringList if x.startswith('rachel:')]
    # filter our words without much semantic meaning
    filtered_out = filter_words(filtered)
    # create chunks of 30 sentences at a time
    chunks_filtered = []
    for index in range(0, len(filtered_out), 30):
        chunks_filtered.append(' '.join(filtered_out[index:index+30]))
    
    LDA(chunks_filtered, "topics_rachel.csv", "rachel_vis.html")
    
    
    
    
    print("[INFO] running Phoebe´s model ...")
    
    # filtering only Phoebe´s lines
    filtered = [x for x in stringList if x.startswith('phoebe:')]
    # filter our words without much semantic meaning
    filtered_out = filter_words(filtered)
    # create chunks of 30 sentences at a time
    chunks_filtered = []
    for index in range(0, len(filtered_out), 30):
        chunks_filtered.append(' '.join(filtered_out[index:index+30]))
    
    LDA(chunks_filtered, "topics_phoebe.csv", "phoebe_vis.html")
    
    
    
    print("[INFO] running Joey´s model ...")
    
    # filtering only Joey´s lines
    filtered = [x for x in stringList if x.startswith('joey:')]
    # filter our words without much semantic meaning
    filtered_out = filter_words(filtered)
    # create chunks of 30 sentences at a time
    chunks_filtered = []
    for index in range(0, len(filtered_out), 30):
        chunks_filtered.append(' '.join(filtered_out[index:index+30]))
    
    
    LDA(chunks_filtered, "topics_joey.csv", "joey_vis.html")
    
    
    
    """------ ------
    """
    

    print("Script was executed successfully! Have a nice day")

    
         
# Define behaviour when called from command line
if __name__=="__main__":
    main()

