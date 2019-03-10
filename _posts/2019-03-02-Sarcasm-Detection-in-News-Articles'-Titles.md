---
title: Sarcasm-Detection-in-News-Articles'-Titles
date: "2 Mar 2019"
---

Just one of the times you stumble upon an excellent dataset on Kaggle for a really interesting data mining problem - sarcasm detection in text and cannot resist to take a stab at it. I have looked for labelled datasets for this problem earlier but couldn't find a reasonably clean corpus with sufficient instances.

But this json has a class-balanced dataset with ~27K news headlines labelled as sarcastic or non-sarcastic.
_**[Kaggle Link to Dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)**_

This weekend data mining endeavour has been a good exercise to make fun discoveries around what makes a headline to be sarcastic.
Some discoveries were quite specific to this dataset. For example - I was surprised when the words 'Area' and 'Man' appeared in my top 10 features to identify sarcasm in news headlines. But then I found out _**['Area Man' is a sarcastic slang used as recurring joke on theonion.com](https://forum.wordreference.com/threads/what-is-an-area-man-or-area-woman.2534374/)**_

```python
raw_df[tokenDataFrame_Final.area == 1][['article_link','headline_feature','is_sarcastic']].head(3).reset_index(drop=True)
```

|article_link|headline_feature|is_sarcastic| 
| -------------| -------------| -------------| 
|https://local.theonion.com/area-woman-said-sorry-118-times-yesterday-1819576089|area woman said 'sorry' 118 times yesterday|1|
|https://www.theonion.com/area-insurance-salesman-celebrates-14th-year-of-quoting-1819565058|area insurance salesman celebrates 14th year of quoting fletch|1|
|https://local.theonion.com/is-area-man-going-to-finish-those-fries-1819565422|is area man going to finish those fries?|1|

Whereas few discoveries are generalized and appear in sarcastic text everywhere and even corroborate with my personal experience. For example - _**'Clearly'**_ popped up in top 10 features. And if you think of it people do tend to use the word frequently in sarcastic remarks.

|article_link|headline_feature|is_sarcastic|
| -------------| -------------| -------------|
|https://www.theonion.com/jealous-gps-clearly-wants-man-to-back-over-wife-1819589581|jealous gps clearly wants man to back over wife|1|
|https://politics.theonion.com/new-job-posting-on-craigslist-clearly-for-secretary-of-1819568699|new job posting on craigslist clearly for secretary of the interior|1|
|https://www.theonion.com/elementary-schooler-clearly-just-learned-to-swear-1819566113|elementary schooler clearly just learned to swear|1|


Cool... now, let's get down to step-by-step going about the problem - Data Cleaning & Exploration, Feature Engineering & Model Training/Testing  

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

_Note - Creating lists with List Comprehensions is more concise and significantly faster than defining functions with For Loops because it [escapes calling append attribute of the list in each iteration](https://stackoverflow.com/questions/30245397/why-is-a-list-comprehension-so-much-faster-than-appending-to-a-list). Hence, have resorted to LCs everywhere._

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
_Sample Term Frequency dataset using two instances to demonstrate_ 

 
|headline&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|better|black|catch|clerk|code|former|minority|mood|political|revival|roseanne|secret|shopper|store|sue|thorny|versace|worse| 
| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------| -------------|	
|former versace store clerk sue secret black code minority shopper|0|1|0|1|1|1|1|0|0|0|0|1|1|1|1|0|1|0|
|roseanne revival catch thorny political mood better worse|1|0|1|0|0|0|0|1|1|1|1|0|0|0|0|1|0|1|


The unigram BoW models ignore the word order and context. We can resort to n-grams to retain some context but it will lead to sparse high dimensional feature vectors. (have therefore commented the code chunk for n-grams)


**Semantics**

A better way to capture the context and latent relationships between different words including synonyms, antonyms, anologies etc is to use word2vec. The word2vec unsupervised models can learn context through vector representation of words called 'word embeddings' based on conditional probabilities of word occurrences around other words. Surprisingly, even with low dimensionality in hundreds, word2vec embeddings can learn really meaningful relationships. Once we have vectors for words, we can take the mean or sum of all words in a document to represent whole document as a single vector. There are also alternate doc2vec methods which directly learn vector representation for documents. I have chosen to average the word2vec embeddings because based on many posts on [stack overflow community](https://stackoverflow.com/questions/45234310/doc2vec-worse-than-mean-or-sum-of-word2vec-vectors) they perform better than doc2vec when dealling with small to medium size corpus.

```python
import gensim
w2v_size = 100
model = gensim.models.Word2Vec(raw_df['tokens'], size = w2v_size)
w2v = dict(zip(model.wv.index2word, model.wv.syn0))

# Word2Vec of Words
fetch_w2v = lambda x : [w2v[w] for w in x if w in w2v]
mean_w2v = lambda x : np.sum(x, axis=0)/len(x)
raw_df['fetch_w2v'] = raw_df['tokens'].apply(fetch_w2v)
raw_df['mean_w2v'] = raw_df['fetch_w2v'].apply(mean_w2v)
```
There is some more data wrangling then to transform the array of vectors to dataframe with each vector as a row. Corresponding code in Jupyter Notebook.

Sample 2-D representation of the word embeddings to show the model works.

```python
# Principal Component Analysis to represent word embeddings in 2-D
from sklearn.decomposition import PCA
from matplotlib import pyplot
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
```

```python
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))

pyplot.rcParams['figure.figsize'] = [50, 100]
pyplot.show()
```

<img src="/assets/images/word2vec_visualization.PNG" alt="word2vec_visualization" class="inline"/>

The behind the scenes magic of word2vec worked amazingly well as the model is grouping words by meaningful relationships:
* Hillary-Clinton-Donald-Trump
* Obama-President-White-House-Campaign
* Area-Man




