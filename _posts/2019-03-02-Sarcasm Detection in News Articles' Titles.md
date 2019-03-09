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


The news articles from theonion.com are all sarcastic whereas the ones from huffingpost.com are all non-Sarcastic. Since, the intention is to understand the linguistic features - vocabulary or semantics that help us identify sarcasm than building a 100% accurate model using just the website_name as a feature, we'll not use this variable while building our model.
```python
pd.pivot_table(raw_df, values=['is_sarcastic'], index=['website_name'], #columns=['is_sarcastic'], 
               aggfunc=('sum','count'), fill_value=0)
```

|website_name|is_sarcastic - count|is_sarcastic - sum|		
| -------------| -------------| -------------|	
|entertainment.theonion.com|1194|1194|
|local.theonion.com|2852|2852|
|politics.theonion.com|1767|1767|
|sports.theonion.com|100|100|
|www.huffingtonpost.com|14985|0|
|www.theonion.com|5811|5811|






