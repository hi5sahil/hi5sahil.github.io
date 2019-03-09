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
raw_df.head()
```

|article_link|headline|is_sarcastic|
| -------------| -------------| -------------|
|https://www.huffingtonpost.com/entry/versace-black-code_us_5861fbefe4b0de3a08f600d5|former versace store clerk sues over secret 'black code' for minority shoppers|0|
|https://www.huffingtonpost.com/entry/roseanne-revival-review_us_5ab3a497e4b054d118e04365|the 'roseanne' revival catches up to our thorny political mood, for better and worse|0|
|https://local.theonion.com/mom-starting-to-fear-son-s-web-series-closest-thing-she-1819576697|mom starting to fear son's web series closest thing she will have to grandchild|1|
|https://politics.theonion.com/boehner-just-wants-wife-to-listen-not-come-up-with-alt-1819574302|boehner just wants wife to listen, not come up with alternative debt-reduction ideas|1|
|https://www.huffingtonpost.com/entry/jk-rowling-wishes-snape-happy-birthday_us_569117c4e4b0cad15e64fdcb|j.k. rowling wishes snape happy birthday in the most magical way|0|











