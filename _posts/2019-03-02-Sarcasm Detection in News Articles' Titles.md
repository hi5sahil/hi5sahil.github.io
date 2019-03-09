---
title: Sarcasm-Detection-in-News-Articles'-Titles
date: "2 Nov 2019"
---

Just one of the times you stumble upon an excellent dataset on Kaggle for a really interesting data mining problem - sarcasm detection in text. I have looked for labelled datasets for this problem earlier but couldn't find a reasonably clean corpus with sufficient instances.

But this json has a class-balanced dataset with ~27K news headlines labelled as sarcastic or non-sarcastic.
[Kaggle Link to Dataset](https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home)

```python
# Reading the JSON File
raw_df = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

# Extracting the Hostname from URL using regular expressions
raw_df['website_name'] = raw_df['article_link'].str.extract('(https://.*?[.]comhttp/'
                                                            '|https://.*?[.]com)', expand=True)
raw_df['website_name'] = raw_df['website_name'].str.replace('https://','').str.replace('/','').str.replace('comhttp','com')
#raw_df = raw_df.drop(['article_link'], axis=1)
raw_df.head(3)
```

|article_link|headline|is_sarcastic|website_name|
| -------------| -------------| -------------| -------------|
|https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5|former versace store clerk sues over secret 'black code' for minority shoppers|0|www.huffingtonpost.com|
|https://www.huffingtonpost.com/entry/roseanne-revival-review_us_5ab3a497e4b054d118e04365|the 'roseanne' revival catches up to our thorny political mood, for better and worse|0|www.huffingtonpost.com|
|https://local.theonion.com/mom-starting-to-fear-son-s-web-series-closest-thing-she-1819576697|mom starting to fear son's web series closest thing she will have to grandchild|1|local.theonion.com|










