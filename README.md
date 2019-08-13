# tweet-sentiment-analysis

Sentiment analysis of tweets using the [Sentiment140](http://help.sentiment140.com/for-students) dataset. The analysis consisted
of basic preprocessing (tokenization, stemming, etc...), data visualization, feature extraction (Bag-of-Words and TF-IDF), and
model training/evaluation.

## Dataset

The Sentiment140 training dataset consists of 1.6 million tweets of negative and positive sentiment. There are six fields for each
data instance: label, id, date, query, user, and text. A preview of the data is shown below:

![alt text](https://github.com/BrennoR/tweet-sentiment-analysis/blob/master/plots/data_preview.PNG "Data Preview")

The id, date, query, and user fields where removed as they add no valuable information to the analysis. Tweet text and sentiment label
were therefore the only fields used.

## Exploration

Very basic exploration ws performed on the dataset in order to get a better understanding of word counts. The top 20 most common
words are plotted below. As shown, the most common words consist of pronouns, prepositions, and small words that are a requirement in
any sentence. Removing these words at a later stage may prove to be beneficial.

![alt text](https://github.com/BrennoR/tweet-sentiment-analysis/blob/master/plots/20_most_cmn_words.PNG "20 Most Common Words")

## Preprocessing

Preprocessing consisted of three steps: basic stripping of the tweets, tokenization, and stemming. The stripping involved conversion
to lowercase and the removal of punctuation, special characters, html links, @user handles, and words with a length of three characters
or less. The next step, tokenization, split the tweets into a group of words (tokens). Finally, stemming reduced inflected word forms
into their stems. Performing this preprocessing prior to any analysis is absolutely vital.

## Visualization

The most common words present in negative and positive tweets were then visualized using wordclouds. Wordclouds are a very effective
visualization tool that displays present words with their size being relative to their count. The two wordclouds are shown below:

![alt text](https://github.com/BrennoR/tweet-sentiment-analysis/blob/master/plots/neg_wordcloud.PNG "Negative Wordcloud")

![alt text](https://github.com/BrennoR/tweet-sentiment-analysis/blob/master/plots/pos_wordcloud.PNG "Positive Wordcloud")

## Feature Extraction

Two forms of feature extraction were performed prior to training, Bag-of-Words using Scikit-Learn's CountVectorizer and TF-IDF (Term
frequency - inverse document frequency) using Scikit-Learn's TfidfVectorizer.

## Model Training

Four algorithms have been used to train on the data: Multinomial Naive Bayes, Logistic Regression, Decision Tree, and Random Forest.
Each algorithm was trained on both a Bag-of-Words and TF-IDF representation. The accuracies achieved using the default parameters for
each model is shown in the table below:

| Model                   | Representation | Accuracy |
|-------------------------|----------------|----------|
| Multinomial Naive Bayes | Bag-of-Words   | 74.69%   |
| Multinomial Naive Bayes | TF-IDF         | 74.30%   |
| Logistic Regression     | Bag-of-Words   | 75.55%   |
| Logistic Regression     | TF-IDF         | 75.82%   |
| Decision Tree           | Bag-of-Words   | 69.52%   |
| Decision Tree           | TF-IDF         | 69.94%   |
| Random Forest           | Bag-of-Words   | 72.96%   |
| Random Forest           | TF-IDF         | 73.07%   |

As shown above, a combination of Logistic Regression with TF-IDF representation performed the best out of all models with a validation
accuracy of 75.82%. As it performed the best, this combo was then chosen for fine tuning.

## Model Fine Tuning

Fine tuning was then performed on the best model from the training step (Logistic Regression, TF-IDF). Randomized search was used to
test a variety of hyperparameters and evaluate their accuracy. The parameter distribution used for the search was: 'penalty' ['l1', 'l2'],
'C': np.logspace(0, 5, 10). The best estimator achieved an accuracy of 75.59%. This is lower than what was achieved with the default
hyperparameters. As such, the default hyperparameters for Logistic Regression were shown to be more than adequate.

## Future Work

There is much work to be done on this dataset. Many different models such as SVM and deep learning architectures like RNNs must be tested
and evaluated. In addition there are other feature representations that can be used such as Word2Vec.
