# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os
import socket
from tqdm import tqdm

# libraries to parallelize code
import multiprocessing as mp
from joblib import Parallel, delayed
import matplotlib.pyplot as plt


# %%
path_training= "./data/training_data.csv"

df = pd.read_csv(path_training, error_bad_lines=False,encoding='utf-8',sep="\t")


# %%
x = df[["x_id","x_name"]]
y = df[['y_id','y_name',"eval"]]
del df['eval']


# %%
x_final = x.drop_duplicates()
y_final = y.drop_duplicates()


# %%
from sklearn.feature_extraction.text import TfidfVectorizer
#ngram best result

import re
import time
from ftfy import fix_text #  text c<leaning for decode issues..
from scipy.sparse import csr_matrix # may not be required 
from scipy.sparse import rand # may not be required
import nmslib


# %%
def ngrams(string, n=4):
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


# %%
article_name = list(y_final['y_name'][:].unique())
vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
tf_idf_matrix = vectorizer.fit_transform(article_name)

# messy data => EX: popsike data
messy_tracks = list(x_final['x_name'][:].dropna().unique())
messy_tf_idf_matrix = vectorizer.transform(messy_tracks)


# %%
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

# c'est là que ça prend le plus de temps
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
  origional_nm = messy_tracks[i] #changer les noms mais du coup le original c'est bien musicbrainz qui est le original et popsike le atched name
  try:
    matched_nm   = article_name[nbrs[i][0][0]]
    conf         = nbrs[i][1][0]
  except:
    matched_nm   = "no match found"
    conf         = None
  mts.append([origional_nm,matched_nm,conf])

mts = pd.DataFrame(mts,columns=['origional_name','matched_name','conf'])
mts['conf'] = -mts['conf'] #otherwise -1 represent confidence = 100%
#results = df_CF.merge(mts,left_on='buyer',right_on='origional_name')
print(mts)
mts = mts.sort_values(["conf"], ascending=False)


# %%
df_test = mts.merge(x_final, how="left",left_on="origional_name", right_on="x_name")
df_test = df_test.merge(y_final, how="left",left_on="matched_name", right_on="y_name")
del df_test['x_name']
del df_test['y_name']


# %%
df_final = df_test.merge(df, how="left",left_on="x_id", right_on="x_id")
df_final = df_final.drop(columns=['set_ratio','sort_ratio','ratio','overlap','soundex','score','max','reverse'])
df_final['eval_fuzz'] = [1 if s >=0.5 else 0 for s in df_final['conf']]
#eval column


# %%
# problem with this algos
cuteoff = np.arange(0,1.05,0.05)
#cuteoff = [0.4,0.5,0.6,0.7,0.8]
list_tpr = []
list_fpr = []
list_accuracy = []
for j in cuteoff:
    print(f'cuteoff: {j}')
    df_final = df_test.merge(df, how="left",left_on="x_id", right_on="x_id")
    df_final = df_final.drop(columns=['set_ratio','sort_ratio','ratio','overlap','soundex','score','max','reverse'])
    df_final['eval_fuzz'] = [1 if s >=j else 0 for s in df_final['conf']]
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(df_final[:])):
        if df_final['eval_fuzz'].loc[i] == 1 and df_final['eval'].loc[i] == 1:
            TP += 1
        elif df_final['eval_fuzz'].loc[i] == 0 and df_final['eval'].loc[i] == 0:
            TN += 1
        elif df_final['eval_fuzz'].loc[i] == 0 and df_final['eval'].loc[i] == 1:
            FN += 1
        else:
            FP += 1

    TPR = TP / (TP+FN)
    FPR = FP / (TN+FP)
    list_tpr.append(TPR)
    list_fpr.append(FPR)
    accuracy_score = round(TP/len(df_final[:]),4)
    list_accuracy.append(accuracy_score)
    print(f'Accuracy score: {accuracy_score*100}%') 
    #print(f'TPR: {TPR}')
    #print(f'FPR: {FPR}')

    #print(f'TP score: {TP}')
    #print(f'FP score: {FP}')
    #print(f'TN score: {TN}')
    #print(f'FN score: {FN}')

# accuracy score vs cuteoff
fig = plt.figure()
plt.plot(cuteoff,list_accuracy)
plt.grid(True)
plt.title('Accuracy score vs cuteoff')
plt.xlabel('cuteoff')
plt.ylabel('Accuracy score')
#plt.show()
fig.savefig('plots/accuracy_vs_cuteoff.png')

#FPR vs cuteoff
fig = plt.figure()
plt.plot(cuteoff,list_fpr)
plt.grid(True)
plt.title('FPR vs cuteoff')
plt.xlabel('cuteoff')
plt.ylabel('FPR')
#plt.show()
fig.savefig('plots/FPR_vs_cuteoff.png')

# TPR vs cuteoffs
fig = plt.figure()
plt.plot(cuteoff,list_fpr)
plt.grid(True)
plt.title('TPR vs cuteoff')
plt.xlabel('cuteoff')
plt.ylabel('TPR')
#plt.show()
fig.savefig('plots/TPR_vs_cuteoff.png')

#ROC curve
line = np.arange(0,1.1,0.1)
fig = plt.figure()
fprs = list_fpr
fprs.append(0)
tprs = list_tpr
tprs.append(0)
plt.plot(fprs,tprs)
plt.plot(line,line)
plt.grid(True)
plt.title('ROC curve')
plt.xlabel('FPR')
plt.ylabel('TPR')
#plt.show()
fig.savefig('plots/fuzzymatching_roccurve.png')




