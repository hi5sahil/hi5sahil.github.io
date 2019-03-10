---
title: Sarcasm-Detection-in-News-Articles'-Titles
date: "2 Mar 2019"
---

Just one of the times you stumble upon an excellent dataset on Kaggle for a really interesting data mining problem - sarcasm detection in text. I have looked for labelled datasets for this problem earlier but couldn't find a reasonably clean corpus with sufficient instances.

But this json has a class-balanced dataset with ~27K news headlines labelled as sarcastic or non-sarcastic.
[Kaggle Link to Dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)

_**Sample Data Exhibit**_

```python
# Reading the JSON File
raw_df = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

# Extracting the Hostname from URL using regular expressions
raw_df['website_name'] = raw_df['article_link'].str.extract('(https://.*?[.]comhttp/'
                                                            '|https://.*?[.]com)', expand=True)
raw_df['website_name'] = raw_df['website_name'].str.replace('https://','').str.replace('/','').str.replace('comhttp','com')
raw_df = raw_df.drop(['article_link'], axis=1)
raw_df.head(3)
```

|headline|is_sarcastic|website_name|
| -------------| -------------| -------------|
|former versace store clerk sues over secret 'black code' for minority shoppers|0|www.huffingtonpost.com|
|the 'roseanne' revival catches up to our thorny political mood, for better and worse|0|www.huffingtonpost.com|
|mom starting to fear son's web series closest thing she will have to grandchild|1|local.theonion.com|


The news articles from theonion are all sarcastic whereas the ones from huffingpost are all non-sarcastic. Since, the aim is to understand the linguistic features - vocabulary or semantics that help us identify sarcasm rater than building a 100% accurate model using just the website_name as a feature, we'll not use this variable for modelling.

```python
pd.pivot_table(raw_df, values=['is_sarcastic'], index=['website_name'], aggfunc=('sum','count'), fill_value=0)
```

|website_name|is_sarcastic - count|is_sarcastic - sum|		
| -------------| -------------| -------------|	
|entertainment.theonion.com|1194|1194|
|local.theonion.com|2852|2852|
|politics.theonion.com|1767|1767|
|sports.theonion.com|100|100|
|www.huffingtonpost.com|14985|0|
|www.theonion.com|5811|5811|


_**Data Cleaning**_

For NLP algorithms - Bag of Words or Doc2Vec, we'll first need a clean set of tokens.
Note - Creating lists with _List Comprehensions_ is more concise and significantly faster than defining functions with _For Loops_ because it [escapes calling append attribute of the list in each iteration](https://stackoverflow.com/questions/30245397/why-is-a-list-comprehension-so-much-faster-than-appending-to-a-list). Hence, have resorted to LCs everywhere. 

Ok, let's first start by some standard data cleaning steps while working with text:

**1. Tokenizing**
```python
# Split into Words
from nltk.tokenize import word_tokenize
raw_df['tokens'] = raw_df['headline_feature'].apply(nltk.word_tokenize)
```

**2. Normalizing Case**
```python
# Convert to lower case
lower_case_tokens = lambda x : [w.lower() for w in x]
raw_df['tokens'] = raw_df['tokens'].apply(lower_case_tokens)
```

**3. Removing Punctuation**
```python
# Filter Out Punctuation
import string
punctuation_dict = str.maketrans(dict.fromkeys(string.punctuation))
# This creates a dictionary mapping of every character from string.punctuation to None

punctuation_remover = lambda x : [w.translate(punctuation_dict) for w in x]
raw_df['tokens'] = raw_df['tokens'].apply(punctuation_remover)
```

**4. Removing Non-alphabetic Tokens**
```python
# Remove remaining tokens that are not alphabetic
nonalphabet_remover = lambda x : [w for w in x if w.isalpha()]
raw_df['tokens'] = raw_df['tokens'].apply(nonalphabet_remover)
```

**5. Filtering out Stop Words**
```python
# Filter out Stop Words
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
stopwords_remover = lambda x : [w for w in x if not w in stop_words]
raw_df['tokens'] = raw_df['tokens'].apply(stopwords_remover)
```

**6. Stemming / Lemmatizing the Tokens**
```python
# Stem / Lemmatize the Words
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
word_lematizer = lambda x : [lmtzr.lemmatize(w) for w in x]
raw_df['tokens'] = raw_df['tokens'].apply(word_lematizer)
```

_**Feature Engineering**_

**Bag of Words**

We can extract Term Frequency to build a Bag of Words model or else use TFIDF statistic (which discounts words which are too common across documents). Though instead of calculating the TFIDF statistic, I chose to simple remove terms which are too common or too rare because removing redundant features all together seems preferrable to avoid over-fitting arising from high dimensionality.

Note - The earlier cleaning steps related to Stop Words removal and Non-Alphabetic tokens removal also addressed redundant dimensions.

```python
sentence_creator = lambda x : [' '.join(x)][0]
raw_df['sentence_feature'] = raw_df['tokens'].apply(sentence_creator)
```

```python
import sklearn.feature_extraction.text as sfText

vect = sfText.CountVectorizer()#(ngram_range = (1, 2))
vect.fit(raw_df['sentence_feature'])

X = vect.transform(raw_df['sentence_feature'])
tokenDataFrame = pd.DataFrame(X.A, columns = vect.get_feature_names())
tokens_redundant = token_sums[token_sums < 10].index
tokenDataFrame2 = tokenDataFrame.drop(tokens_redundant, axis = 1)
```
_Sample Term Frequency dataset using two headlines_ 

|headline&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|better|black|catch|clerk|code|former|minority|mood|political|revival|roseanne|secret|shopper|store|sue|thorny|versace|worse| 
| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------|	
|former versace store clerk sue secret black code minority shopper|0|1|0|1|1|1|1|0|0|0|0|1|1|1|1|0|1|0|
|roseanne revival catch thorny political mood better worse|1|0|1|0|0|0|0|1|1|1|1|0|0|0|0|1|0|1|

The unigram BoW models ignore the word order and context. We can resort to n-grams but it will lead to sparse high dimensional feature vectors. (have commented the code chunk for n-grams)


**Semantics**



