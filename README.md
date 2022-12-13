# Word-Embedding-Modelling-and-Applications
Implements complete Word2Vec (Skipgram, CBOW, Glove) for modelling word embedding and uses them for a variety of Real-World Applications such as Named-Entity Recognition and Neural Dependency Parsing

# Word2Vec Implementation and Modelling

## Introduction

Word2Vec is a group of related models that are used to produce word embeddings. These models are shallow, two-layer neural networks that are trained to reconstruct linguistic contexts of words. The idea is that words that share contexts in the corpus will be located in close proximity to one another in the word vector space.

We have implemented 3 models from scratch -

1. **Continuous Bag of Words (CBOW) (With and Without Negative Sampling):** The CBOW model predicts a target word (the center word) from the source context words (surrounding words).
2. **Skip-Gram (With and Without Negative Sampling):** The Skip-Gram model predicts source context words from the target word.
3. **GloVe (Global Vectors for Word Representation) (With and Without Negative Sampling):** Glove is a model that is trained on a global word-word co-occurrence matrix, which is constructed from a corpus. The co-occurrence matrix is a sparse matrix that contains the number of times each word occurs in the context of each other word. The GloVe model is trained to predict the co-occurrence probabilities from the word vectors. https://github.com/stanfordnlp/GloVe

## Dataset

### For Word2Vec Training

We have used the Latest Wikipedia Dump with 100 GB of textual data. The dataset is available at https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

### For Testing

SimLex-999: A Human Judged Semantic Similarity Dataset for Evaluating Word Embeddings - https://fh295.github.io/simlex.html

## Model Parameters

1. **Embedding Dimension - 300**
2. **Window Size - 5**
3. **Learning Rate - 0.01**
4. **Batch Size - 100**
5. **Epochs - 10**
6. **Negative Sampling - 5**

## Testing

We have used the SimLex-999 dataset for testing the word embeddings. The dataset contains 999 word pairs with human-annotated similarity scores. The dataset is available at https://fh295.github.io/simlex.html

We have calculated the Spearman's Rank Correlation Coefficient between the human-annotated similarity scores and the cosine similarity between the word embeddings of the word pairs.

## Visualizing the Word Embeddings

We have visualized the behaviour of the word embeddings using 10 semantic word pairs.

The word pairs are -

1. **man** is to **woman** as **son** is to **daughter**
2. **fast** is to **fastest** as **cold** is to **coldest**
3. **large** is to **largest** as **high** is to **highest**
4. **sing** is to **singing** as **walk** is to **walking**
5. **predicting** is to **predicted** as **flying** is to **flew**
6. **saying** is to **said** as **implementing** is to **implemented**
7. **buy** is to **bought** as **sell** is to **sold**
8. **seeing** is to **saw** as **singing** is to **sang**
9. **walking** is to **walked** as **sleeping** is to **slept**
10. **thinking** is to **thought** as **flying** is to **flew**

The word embeddings are visualized using t-SNE.

## Results

### Skip-Gram without Negative Sampling

![image](https://user-images.githubusercontent.com/63910248/207248009-0a175347-ccfc-4563-acd9-98c4d896ec91.png)


### Skip-Gram with Negative Sampling

![image](https://user-images.githubusercontent.com/63910248/207248104-b41c73d0-f68b-4643-9e4d-97c5007096b4.png)


### CBOW without Negative Sampling

![image](https://user-images.githubusercontent.com/63910248/207248172-38e7035d-c68c-4b1a-a2f8-2ae1bee5e1de.png)


### CBOW with Negative Sampling

![image](https://user-images.githubusercontent.com/63910248/207248244-8ffa3a85-c76e-475b-b4fe-9c2583c4845c.png)


### GloVe without Negative Sampling

![image](https://user-images.githubusercontent.com/63910248/207248289-b2a45679-855c-469a-b5ce-bc23c6d95684.png)


### GloVe with Negative Sampling

![image](https://user-images.githubusercontent.com/63910248/207248357-ca38511c-61f0-4a70-98b6-031293a95a26.png)
