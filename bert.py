import pandas as pd
import numpy as np

import re

from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

import gensim
from gensim import corpora, models, similarities
from gensim.models import CoherenceModel, HdpModel
#from gensim.models.nmf import Nmf

from collections import OrderedDict

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from collections import Counter

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

## Data Processing
data_file = "dreaddit-train-allTexts.csv"
train_df = pd.read_csv(data_file)
# train_df.head()

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
                         "it", "make", "but" , "my"]
    stopwords_list = stops + selfdefined_stops
    # stem
    if stem:
        stemmer = PorterStemmer()
        toks = [stemmer.stem(tok) for tok in toks]
    # remove punctuation
    toks_nopunc = [tok.lower() for tok in toks if tok not in string.punctuation]
    # remove stopwords
    toks_nostop = [tok for tok in toks_nopunc if tok not in stopwords_list]
    toks_tidy = [tok for tok in toks_nostop if re.match(r'[a-zA-Z]+', tok) and len(tok) >= 2]
    return toks_tidy

train_df['text'] = train_df.apply(lambda x: preprocess_text(x.text), axis = 1)
#train_df.head()

docs = train_df["text"].str.join(" ")
#docs

## Training
# instantiating BERTopic
topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)
topics, probs = topic_model.fit_transform(docs)

## Extracting Topics
# looking at the results. Typically, we look at the most frequent topics first as they best represent the collection of documents.
freq = topic_model.get_topic_info()
#freq.head(5)
# -1 refers to all outliers and should typically be ignored

#   frequent topic that were generated:
topic_model.get_topic(0)  # Select the most frequent topic

#### Visualization
topic_model.visualize_topics()

## Visualize Topic Probabilities
topic_model.visualize_distribution(probs[10], min_probability=0.01)

## Visualize Topic Hierarchy
topic_model.visualize_hierarchy()

### Visualize Terms
topic_model.visualize_barchart(top_n_topics=3)


######## Topic Reduction to 3
new_topics, new_probs = topic_model.reduce_topics(docs, topics, probs, nr_topics=3)

## Extracting Topics
freq_new = topic_model.get_topic_info()
#freq_new.head(5)

#### Visualization
topic_model.visualize_topics()
topic_model.visualize_distribution(probs[10], min_probability=0,width=700, height=400)
topic_model.visualize_hierarchy(width=700, height=1000)
topic_model.visualize_barchart()
topic_model.get_topic(0)

###### Dimension reductio
model_bert = SentenceTransformer('bert-base-nli-max-tokens')

embedding_bert = np.array(model_bert.encode(docs, show_progress_bar=True))
#Bert embeddings are shape of 768
print("Bert Embedding shape", embedding_bert.shape)
print("Bert Embedding sample", embedding_bert[0][0:50])

def predict_topics_with_kmeans(embeddings,num_topics):
  kmeans_model = KMeans(num_topics)
  kmeans_model.fit(embeddings)
  topics_labels = kmeans_model.predict(embeddings)
  return topics_labels


def plot_embeddings(embedding, labels,title):

    labels = np.array( labels )
    distinct_labels =  set( labels )
    
    n = len(embedding)
    counter = Counter(labels)
    for i in range(len( distinct_labels )):
        ratio = (counter[i] / n )* 100
        cluster_label = f"cluster {i}: { round(ratio,2)}"
        x = embedding[:, 0][labels == i]
        y = embedding[:, 1][labels == i]
        plt.plot(x, y, '.', alpha=0.4, label= cluster_label)
    plt.legend(title="Topic",loc = 'upper left', bbox_to_anchor=(1.01,1))
    plt.title(title)
    

def reduce_pca(embedding):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform( embedding )
    print( "pca explained_variance_ ",pca.explained_variance_)
    print( "pca explained_variance_ratio_ ",pca.explained_variance_ratio_)
    
    return reduced


def reduce_tsne(embedding):
    tsne = TSNE(n_components=2)
    reduced = tsne.fit_transform( embedding )
    
    return reduced

num_topics = 3
#Apply Kmeans without dimension reduction
labels_bert_raw  = predict_topics_with_kmeans(embedding_bert,num_topics)

#Apply Kmeans for Bert Vectors  with PCA  dimension reduction
embedding_bert_pca =  reduce_pca( embedding_bert )
labels_bert_pca  = predict_topics_with_kmeans(embedding_bert_pca,num_topics)
plot_embeddings(embedding_bert_pca,labels_bert_pca,"Bert with PCA")

#Apply Kmeans for Bert Vectors  with T-sne  dimension reduction
embedding_bert_tsne =  reduce_tsne( embedding_bert )
labels_bert_tsne  = predict_topics_with_kmeans(embedding_bert_tsne,num_topics)
plot_embeddings(embedding_bert_tsne,labels_bert_tsne,"Bert with T-sne")


print("Silhouette score:" )
print("Raw Bert" ,silhouette_score(embedding_bert, labels_bert_raw) )
print("Bert with PCA" ,  silhouette_score(embedding_bert_pca, labels_bert_pca) )
print("Bert with Tsne" , silhouette_score(embedding_bert_tsne, labels_bert_tsne) )




