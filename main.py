import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import re

from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, HdpModel
from gensim.models.nmf import Nmf

from collections import OrderedDict

import pyLDAvis
import pickle

from collections import Counter
import matplotlib.colors as mcolors


def getDomainsfromTopics(topic):
    if topic in ['domesticviolence', 'survivorsofabuse']:
        return 'abuse'
    elif topic in ['anxiety', 'stress']:
        return 'anxiety'
    elif topic in ['almosthomeless', 'assistance', 'food_pantry', 'homeless']:
        return 'financial'
    elif topic == 'ptsd':
        return 'PTSD'
    elif topic == 'relationships':
        return 'social'


def getSentiLabelFromScores(score):
    if score > 0:
        return 'Positive'
    elif score < 0:
        return 'Negative'
    else:
        return 'Neutral'


def manual_stem(tok):
    if tok in ['trying', 'tried', 'tries']:
        return 'try'
    if tok in ['getting', 'got', 'gets']:
        return 'get'
    if tok in ['made', 'making', 'makes']:
        return 'make'
    if tok in ['took', 'taking', 'takes']:
        return 'take'
    if tok in ['wanted', 'wants']:
        return 'want'
    if tok in ['goes', 'went', 'going']:
        return 'go'
    if tok in ['told', 'telling', 'tells']:
        return 'tell'
    if tok in ['feeling', 'felt', 'feels', 'feelings']:
        return 'feel'
    if tok in ['found', 'finds']:
        return 'find'
    if tok in ['asked', 'asks', 'asking']:
        return 'ask'
    if tok in ['seems', 'seemed']:
        return 'seem'
    if tok in ['working', 'worked', 'works']:
        return 'work'
    if tok in ['starting', 'started', 'starts']:
        return 'start'
    if tok in ['coming', 'came', 'comes']:
        return 'come'
    if tok in ['best', 'well']:
        return 'good'
    if tok in ['called', 'calls']:
        return 'call'
    if tok in ['thinking', 'thought', 'thinks']:
        return 'think'
    if tok in ['trying', 'tried', 'tries']:
        return 'try'
    if tok == 'years':
        return 'year'
    if tok == 'months':
        return 'month'
    if tok == 'days':
        return 'day'
    if tok == 'friends':
        return 'friend'
    if tok == 'thanks':
        return 'thank'
    if tok == 'jobs':
        return 'job'
    else:
        return tok


# Prepare the corpus for analysis
def preprocess_text(text, stem=False):
    """Preprocess one sentence: tokenizes, lowercases, applies the Porter stemmer,
     removes punctuation tokens and stopwords.
     Returns a list of strings."""
    toks = word_tokenize(text)
    # clean
    stops = stopwords.words('english')
    selfdefined_stops = ["n't", "the", "it", "get", "got", "gets",
                         "my", "also", "one", "could", "would", "can",
                         "as", "said", "go", "goes", "going", "went",
                         "also", "is", "since", "these", "so", "really",
                         "much", "what", "the", "still", "every", "any",
                         "it", "make", "but", "my", "like", "know",
                         "something", "even", "tell", "told", "things",
                         "feel", "want", "think", "take", "back", "never",
                         "first", "last", "ca", "us", "thing", "always",
                         "made", "else", "dont", "im", "find", "though",
                         "let", "way", "come", "ever", "lot", "good", "bad",
                         "maybe", "two", "little", "start", "try", "apparently",
                         "joe", "seem", "give", "please", "next", "able", "put",
                         "although", "etc", "long", "url", "say", "see", "sometimes",
                         "around", "another", "bit", "x200b", "edit", "ask"]
    stopwords_list = stops + selfdefined_stops
    # stem
    if stem:
        stemmer = PorterStemmer()
        toks = [stemmer.stem(tok) for tok in toks]
    toks = [manual_stem(tok) for tok in toks]
    # remove punctuation
    toks_nopunc = [tok.lower() for tok in toks if tok not in string.punctuation]
    # remove stopwords
    toks_nostop = [tok for tok in toks_nopunc if tok not in stopwords_list]
    toks_tidy = [tok for tok in toks_nostop if re.match(r'[a-zA-Z]+', tok) and len(tok) >= 3]
    return toks_tidy


data_file = "./dreaddit/dreaddit-train-allTexts.csv"
train_df = pd.read_csv(data_file)

# EDA
# the distributions of stress topics in the data via a barplot
sns.set(rc={'figure.figsize': (15, 10)})
by_topic = sns.countplot(x='subreddit', data=train_df, order=train_df['subreddit'].value_counts().index)
for item in by_topic.get_xticklabels():
    item.set_rotation(90)
by_topic.set_title("Distribution of Human Annotated Stress Topics")
plt.show()

# Because some subreddits are more or less popular, the amount of data in each domain varies.
# We include ten total subreddits from five domains in our dataset.
train_df['domains'] = train_df.apply(lambda x: getDomainsfromTopics(x.subreddit), axis=1)
stress_domains = list(set(train_df['domains'].tolist()))

# the distributions of re-assigned stress domains in the data via a barplot
sns.set(rc={'figure.figsize': (15, 10)})
by_domain = sns.countplot(x='domains', data=train_df, order=train_df['domains'].value_counts().index)
for item in by_topic.get_xticklabels():
    item.set_rotation(90)
by_domain.set_title("Distribution of Re-assigned Stress Domains")
plt.show()
# now the count witnin each group even
# get sentiment labels
train_df['senti_label'] = train_df.apply(lambda x: getSentiLabelFromScores(x.sentiment), axis=1)

# Take a look at the distributions of human annotated/tagged sentiment by topic via a barplot
senti_labels = list(set(train_df['senti_label'].tolist()))
df_sentiment = train_df.groupby(['domains', 'senti_label'])['domains'].count().unstack('senti_label')
domains_mixture = df_sentiment[senti_labels].plot(kind='bar', stacked=True, legend=True)
domains_mixture.set_title("Distribution of Stress Topics Across Sentiment")
plt.show()
# drop Neutral rows since there are too few records
train_df = train_df.loc[train_df.senti_label != 'Neutral']
# Take a look at the distributions of human annotated/tagged sentiment by topic via a barplot
senti_labels = list(set(train_df['senti_label'].tolist()))
df_sentiment = train_df.groupby(['domains', 'senti_label'])['domains'].count().unstack('senti_label')
domains_mixture = df_sentiment[senti_labels].plot(kind='bar', stacked=True, legend=True)
domains_mixture.set_title("Distribution of Stress Topics Across Sentiment")
plt.show()

# LDA Model
# tidy data
train_df['text'] = train_df.apply(lambda x: preprocess_text(x.text), axis=1)

texts = train_df['text']
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# A major question in using LDA for topic modeling is what is the proper set of
# hyperparmeters to generate the optimal set of topics for the corpus of documents
# under examination. Gensim includes methods for computing the Perplexity and Topic
# Coherence of a corpus. One appraoch is to sample an LDA model for a range of
# for perplexity and topic coherence and select the appropriate number of topics
# from a point of minimum perplexity and maximum topic coherence.

# However, recent studies have shown that predictive likelihood (or equivalently, perplexity) and human judgment are
# often not correlated, and even sometimes slightly anti-correlated.
# So only using coherence score to evaluate models in our project
# perplexity_lda = []
coherence_lda = []
topic_count_lda = []

for num_topics in range(2, 16, 1):
    print("Computing the lda model using {} topics".format(num_topics))
    topic_lda = models.LdaModel(corpus,
                                id2word=dictionary,
                                num_topics=num_topics,
                                iterations=1000,
                                alpha='auto')
    corpus_lda = topic_lda[corpus]  # Use the bow corpus

    topic_count_lda.append(num_topics)

    #     # a measure of how good the model is. the lower, the better.
    #     perplexity_lda.append(topic_lda.log_perplexity(corpus))

    # Compute Coherence Score
    cm = CoherenceModel(model=topic_lda, corpus=corpus, dictionary=dictionary, coherence='u_mass')
    coherence_lda.append(cm.get_coherence())

# Pull the resulting data into a pandas dataframe
topics_lda = pd.DataFrame({'coherence': coherence_lda},
                          index=topic_count_lda)
lines = topics_lda.plot.line()
lines.set_title("Coherence Scores of LDA Models")
plt.show()
# the higher the score is, the better the topic clusters are

# using the number of topics with the highest coherence score
total_topics = 3

# lda model
lda_model = models.LdaModel(corpus, id2word=dictionary, num_topics=total_topics, iterations=1000, alpha='auto')
# Show first n=10 important words in the topics:
lda_model.show_topics(total_topics, 10)
# Load the topic - term data into an python dictionary
data_lda_model = {i: OrderedDict(lda_model.show_topic(i, 10)) for i in range(total_topics)}
print(data_lda_model)

# infer the distribution of topics according to the lda model
topics = []
probs = []
max_to_show = 10

for k, i in enumerate(range(len(texts))):
    try:
        bow = dictionary.doc2bow(texts[i])
        doc_topics = lda_model.get_document_topics(bow, minimum_probability=0.01)
        # topic with the highest probability
        topics_sorted = sorted(doc_topics, key=lambda x: x[0], reverse=True)
        topics.append(topics_sorted[0][0])
        probs.append("{}".format(topics_sorted[0][1]))

        # Dump out the topic and probability assignments for the first 20 texts
        if k < max_to_show:
            print("Text {}: {}".format(k, topics_sorted))
    except KeyError:
        pass

train_df['LDAtopic'] = pd.Series(topics)
train_df['LDAprob'] = pd.Series(probs)

# Resort the dataframe according to the human annotated topic and lda topic
train_df.sort_values(['domains', 'LDAtopic'], ascending=[True, True], inplace=True)

# the distributions of LDA assigned topics in the data via a barplot
sns.set(rc={'figure.figsize': (15, 10)})
by_topic = sns.countplot(x='LDAtopic', data=train_df, order=train_df['LDAtopic'].value_counts().index)
for item in by_topic.get_xticklabels():
    item.set_rotation(90)
by_topic.set_title("Distribution of Topics")
plt.show()

# Resort the dataframe according to the the lda assigned topic and the assocoiated probability
train_df.sort_values(['LDAtopic', 'LDAprob'], ascending=[True, False], inplace=True)
# the topic distrubtions related to the original human annotated/tagged topics
df = train_df.groupby(['LDAtopic', 'domains'])['LDAtopic'].count().unstack('domains')
topic_mixture = df[stress_domains].plot(kind='bar', stacked=True, legend=True)
topic_mixture.set_title("Distribution of LDA-assigned Topics across the Original Human Annotated Topics")
plt.show()

# # Visualize the topics
# pyLDAvis.enable_notebook()
# LDAvis_prepared = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
# LDAvis_prepared

# word count importance of topic keywords
topics = lda_model.show_topics(formatted=False)
data_flat = [w for w_list in texts for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i, weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(1, 3, figsize=(10, 8), sharey=True, dpi=100)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.5, alpha=0.3,
           label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id == i, :], color=cols[i], width=0.2,
                label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030);
    ax.set_ylim(0, 1000)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=12)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id == i, 'word'], rotation=45, horizontalalignment='right')
    ax.legend(loc='upper left');
    ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=18, y=1.05)
plt.show()
