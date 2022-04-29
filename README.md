# ANLY521-FinalProject

## Topic Modeling for Stress Analysis in Social Media
Authors: Yuduo Jin, Yujia Jin, Carmen Wang 

This project intends to apply Latent Dirichlet allocation(LDA) and BERT to detect different stress-related topics using social media data. As in the modern society people rely on the internet for almost everything and usually cope with stress by posting on social media, it would be interesting to explore different topics in their posts. 

## Environment requirements 

A new environment is required to run the BERT model(`Bert_NewEnv.py`) as we wanted to avoid dependency conflicts. Thus, the instruction is provided below:

![WeChat Image_20220428195700](https://user-images.githubusercontent.com/70648734/165865946-2e267e54-ffd1-446e-b4e0-b0f6ae5b5fdd.jpg)

![WeChat Image_20220428195711](https://user-images.githubusercontent.com/70648734/165865982-e6943529-3064-41b6-9fd1-adf2b64a84b9.jpg)

![WeChat Image_20220428195724](https://user-images.githubusercontent.com/70648734/165865996-bed1a24d-20d3-4af4-a4c8-1527793c94b8.jpg)

Next, we need to execute `pip install matplotlib` in the terminal as well to make sure we have all dependencies in place. 


## Files

## `requirements.txt`

Use this to download dependencies that are not in the class environment to run the base LDA model. 

## `dreaddit/dreaddit-train-allTexts.csv`

This is the Dreaddit dataset which inclueds social media posts. 

_Turcan, E., McKeown K. (2019)_ Dreaddit: A Reddit Dataset for Stress Analysis in Social Media. arXiv preprint arXiv:1911.00133 


## `base_LDA.py` 

This is our base LDA model to perform the topic modeling task. 
`getDomainsfromTopics` reassigns topics based on the data label. 
`getSentiLabelFromScores` returns a sentiment label based on the score. 
`manual_stem` manually stems the words in the texts. 
`preprocess_text` is a function that preprocesses the text data, including punctuations removal, tokenization, stop word removal, and stemming. 
`reduce_pca` and `reduce tsne` processes the components and reduce dimensions. 
`get_document_topic_lda` produces the document topic mapping.

usage: 

`python Base_LDA.py --train_file ./dreaddit/dreaddit-train-allTexts.csv`

## `Bert_NewEnv.py`

This is our Bert model to perform the same topic modeling task. 

`predict_topics_with_kmeans` uses k-mean algorithm to predict and return topic labels. 
`plot_embeddings` visualizes the word scatters.
`reduce_pca` and `reduce_tsne` likewise processes the components and reduce dimensions. 
`preprocess_text` is a function that preprocesses the text data, including punctuations removal, tokenization, stop word removal, and stemming. 

usage: 
`python bert.py --train_file ./dreaddit/dreaddit-train-allTexts.csv`


