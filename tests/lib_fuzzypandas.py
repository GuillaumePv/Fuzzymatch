import pandas as pd
import fuzzy_pandas as fpd

# change path
df_x = pd.read_csv('../data/input_x.csv',error_bad_lines=False,encoding='utf-8',sep="\t").iloc[1:,:]
df_y = pd.read_csv('../data/input_y.csv',error_bad_lines=False,encoding='ISO 8859-1',sep="\t",header=None)

df_x['author'] = df_x['author'].astype('string')
df_y['author_y'] = df_y[1].astype('string')

#avoid bug -> take too much time when I increase of the two dataset
# try paralleliez solution => How ?
df_x = df_x.dropna()
df_y = df_y.dropna()
df_x = df_x.iloc[:1000,:]
df_y = df_y.iloc[:1000,:]

matches = fpd.fuzzy_merge(df_x, df_y,
                          left_on=['author'],
                          right_on=['author_y'],
                          ignore_case=True,
                          method='levenshtein',
                          keep='match')