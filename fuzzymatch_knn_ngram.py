## Based on https://colab.research.google.com/drive/1qhBwDRitrgapNhyaHGxCW8uKK5SWJblW

################
## Librairies ##
################

import pandas as pd
import numpy as np
import os
import pickle #optional - for saving outputs => best for large dataset
import re
from tqdm import tqdm # used for progress bars (optional)
import time

from sklearn.feature_extraction.text import TfidfVectorizer
#ngram best result

import re

from ftfy import fix_text #  text cleaning for decode issues..

import nmslib
from scipy.sparse import csr_matrix # may not be required 
from scipy.sparse import rand # may not be required

def ngrams(string, n=3):
    """Takes an input string, cleans it and converts to ngrams. 
    This script is focussed on cleaning UK company names but can be made generic by removing lines below"""
    string = str(string)
    string = string.lower() # lower case
    string = fix_text(string) # fix text
    string = string.split('t/a')[0] # split on 'trading as' and return first name only
    #string = string.split('trading as')[0] # split on 'trading as' and return first name only
    string = string.encode("ascii", errors="ignore").decode() #remove non ascii chars
    chars_to_remove = [")","(",".","|","[","]","{","}","'","-"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']' #remove punc, brackets etc...
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    #string = string.replace('limited', 'ltd')
    #string = string.replace('public limited company', 'plc')
    #string = string.replace('united kingdom', 'uk')
    #string = string.replace('community interest company', 'cic')
    string = string.title() # normalise case - capital at start of each word
    string = re.sub(' +',' ',string).strip() # get rid of multiple spaces and replace with a single
    string = ' '+ string +' ' # pad names for ngrams...
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]

#############
# Import data
#############
df_x = pd.read_csv('./data/input_x.csv',error_bad_lines=False,encoding='utf-8',sep="\t").iloc[1:,:]
df_y = pd.read_csv('./data/input_y.csv',error_bad_lines=False,encoding='ISO 8859-1',sep="\t",header=None)

df_x['author'] = df_x['author'].astype('string')
df_y['author'] = df_y[1].astype('string')

del df_x['id']
del df_y[0]
del df_y[1]

# original data => EX: musicbrainz data
artist_name = list(df_x['author'].dropna().unique())
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(artist_name)

# messy data => EX: popsike data
messy_artist = list(df_y['author'].unique())
messy_tf_idf_matrix = vectorizer.transform(messy_artist)

# create a random matrix to index
data_matrix = tf_idf_matrix#[0:1000000]

# Set index parameters
# These are the most important ones
M = 80
efC = 1000

num_threads = 4 # adjust for the number of threads
# Intitialize the library, specify the space, the type of the vector and add data points 
index = nmslib.init(method='simple_invindx', space='negdotprod_sparse_fast', data_type=nmslib.DataType.SPARSE_VECTOR) 

index.addDataPointBatch(data_matrix)
# Create an index
start = time.time()
index.createIndex() 
end = time.time() 
print('Indexing time = %f' % (end-start))

# Number of neighbors => use Knn
num_threads = 4
K=1
query_matrix = messy_tf_idf_matrix
start = time.time() 
query_qty = query_matrix.shape[0]
nbrs = index.knnQueryBatch(query_matrix, k = K, num_threads = num_threads)
end = time.time() 
print('kNN time total=%f (sec), per query=%f (sec), per query adjusted for thread number=%f (sec)' % 
      (end-start, float(end-start)/query_qty, num_threads*float(end-start)/query_qty))

mts =[]
for i in range(len(nbrs)):
  origional_nm = messy_artist[i]
  try:
    matched_nm   = artist_name[nbrs[i][0][0]]
    conf         = nbrs[i][1][0]
  except:
    matched_nm   = "no match found"
    conf         = None
  mts.append([origional_nm,matched_nm,conf])

mts = pd.DataFrame(mts,columns=['origional_name','matched_name','conf'])
mts['conf'] = -mts['conf'] #otherwise -1 represent confidence = 100%
#results = df_CF.merge(mts,left_on='buyer',right_on='origional_name')
print(mts)