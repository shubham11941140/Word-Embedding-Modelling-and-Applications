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

# Named Entity Recognition and Evaluation

## Introduction

**Named Entity Recognition (NER) is a subtask of Information Extraction that locates and classifies named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc.**

**It is a challenging task in Natural Language Processing (NLP) because of the ambiguity of the language and the presence of many named entities that are not included in the pre-defined categories and the need to generalize to new categories.**

## Dataset

The dataset is available on the following sites. It contains **14041 sentences with 203621 tokens and 4258 named entities.**
```
https://www.clips.uantwerpen.be/conll2003/ner/
https://data.deepai.org/conll2003.zip
https://huggingface.co/datasets/conll2003
```

## Process

We have implemented the **NER model using Word Vectors learned from Skip-Gram, CBOW and GloVe.** We have used the **CoNLL-2003 dataset** for training and testing the model.

Build a binary classification model for each class of the dataset and **predict whether each token is a named entity or not.** We enhance the model by using the word vectors learned from Skip-Gram, CBOW and GloVe. The surrounding words of the token are used as the context for the model.

## Model Parameters

1. **Embedding Dimension - 300**
2. **2-Layer Neural Classifier (Hidden Layer 1- 100 nodes, Hidden Layer 2 - 50 nodes)**
3. **Learning Rate - 0.001**
4. **Batch Size - 100**
5. **Epochs - 10**
6. **Adam Optimizer**
7. **Sigmod Activation Function**
8. **Cross Entropy Loss**

## Testing

Predict the named entities in the test set and evaluate the model using the accuracy, precision, recall and F1 score.

## Results

### Skip-Gram

![image](https://user-images.githubusercontent.com/63910248/207261787-2d34ab9a-c3da-4292-a081-5058fd56cbb9.png)


### CBOW

![image](https://user-images.githubusercontent.com/63910248/207261719-d42e222d-c341-41aa-b5ec-adeb0c93ce8e.png)


### GloVe

![image](https://user-images.githubusercontent.com/63910248/207261840-382132c1-a947-4653-87a5-ca3394729228.png)

# Neural Dependency Parsing and Evaluation

## Introduction

Dependency parsing is a task in Natural Language Processing (NLP) that aims to automatically determine the syntactic structure of a sentence. It is a subtask of parsing that assigns a syntactic dependency structure to a sentence. The syntactic dependency structure is a directed graph that represents the grammatical relations between the words in a sentence. The nodes of the graph are the words in the sentence and the edges are the grammatical relations between the words.

## Dataset

GloVe: Global Vectors for Word Representation - https://nlp.stanford.edu/projects/glove/ - 300 Dimensional Word Vectors used for training the model

## Process

We have used the Universal Dependencies dataset for training and testing the model.

The dataset contains 12543 sentences with 12543 dependency trees. The dataset is available at https://universaldependencies.org/

### Implementation to represent the POS tags and dependency relation types as vectors.

## To Run:

```
    python3 Neural_Transition_Dependency_Parser.py [number of epochs]
```

where,
[number of epochs] - integer argument that specifies the number of epochs for which the model must run

Eg:
```
	python3 Neural_Transition_Dependency_Parser.py 50
```

Ensure the following files are present in the same directory as the python file:

1. **train.conllu** - Training dataset
2. **dev.conllu** - Development dataset
3. **test.conllu** - Testing dataset
4. **glove.6B.300d.txt** - GloVe Word Vectors
5. **en_ewt-ud-test.conllu** - English EWT TreeBank dataset
6. **en_ewt-ud-train.conllu** - English EWT TreeBank dataset
7. **en_ewt-ud-dev.conllu** - English EWT TreeBank dataset
8. **dev_predictions.conllu** - Prediction dataset

## Evaluation

Evaluatation is done using the **UAS (Unlabeled Attachment Score) and LAS (Labeled Attachment Score).**

The English EWT TreeBank is the dataset used for testing the model. The dataset contains 2002 sentences with 2002 dependency trees.

The dataset is available at https://universaldependencies.org/treebanks/en_ewt/index.html

## Output

The output of the first 50 epochs of the model is saved on the output.txt file attached

Result of running the model for 50 epochs:
```
Reading training data
217121it [00:01, 140495.55it/s]
Reading validation data
27150it [00:01, 25825.54it/s]
Generating training instances
100% 12543/12543 [01:18<00:00, 160.76it/s]
2022-10-07 07:11:48.669657: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.

Reading pretrained embedding file.
400000it [00:16, 23882.71it/s]

Generating Training batches:
100% 39/39 [00:03<00:00,  9.79it/s]

Epoch 1 / 50
Average training loss: 11.76 : 100% 39/39 [00:05<00:00,  7.26it/s]
Evaluating validation performance:
100% 2001/2001 [01:23<00:00, 24.00it/s]
UAS: 51.151139210306575
LAS: 41.25014911129667


Epoch 2 / 50
Average training loss: 1.76 : 100% 39/39 [00:02<00:00, 16.10it/s]
Evaluating validation performance:
100% 2001/2001 [01:24<00:00, 23.81it/s]
UAS: 61.099844924251464
LAS: 53.79935583919838


Epoch 3 / 50
Average training loss: 0.84 : 100% 39/39 [00:02<00:00, 15.66it/s]
Evaluating validation performance:
100% 2001/2001 [01:22<00:00, 24.33it/s]
UAS: 64.0104974352857
LAS: 57.68022585391069


Epoch 4 / 50
Average training loss: 0.58 : 100% 39/39 [00:02<00:00, 15.92it/s]
Evaluating validation performance:
100% 2001/2001 [01:23<00:00, 24.02it/s]
UAS: 62.87327527933516
LAS: 57.11559107718001


Epoch 5 / 50
Average training loss: 0.42 : 100% 39/39 [00:02<00:00, 15.66it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.84it/s]
UAS: 68.61903057775658
LAS: 63.064137739075115


Epoch 6 / 50
Average training loss: 0.37 : 100% 39/39 [00:02<00:00, 15.59it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.61it/s]
UAS: 69.13992604079685
LAS: 63.95085291661696


Epoch 7 / 50
Average training loss: 0.25 : 100% 39/39 [00:02<00:00, 15.67it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.83it/s]
UAS: 70.07435683327368
LAS: 64.99264384269752


Epoch 8 / 50
Average training loss: 0.15 : 100% 39/39 [00:02<00:00, 15.90it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.68it/s]
UAS: 69.83975505984333
LAS: 65.11988548252415


Epoch 9 / 50
Average training loss: 0.10 : 100% 39/39 [00:02<00:00, 15.68it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.66it/s]
UAS: 70.77418585232017
LAS: 66.18951051731679


Epoch 10 / 50
Average training loss: 0.08 : 100% 39/39 [00:02<00:00, 15.85it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.91it/s]
UAS: 71.29905761660504
LAS: 66.9291025488091


Epoch 11 / 50
Average training loss: 0.06 : 100% 39/39 [00:02<00:00, 15.59it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.50it/s]
UAS: 71.3070102190942
LAS: 66.996699669967


Epoch 12 / 50
Average training loss: 0.05 : 100% 39/39 [00:02<00:00, 15.83it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 25.01it/s]
UAS: 71.80802417591157
LAS: 67.68857608652432


Epoch 13 / 50
Average training loss: 0.04 : 100% 39/39 [00:02<00:00, 15.95it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.68it/s]
UAS: 71.96707622569485
LAS: 67.74026800270389


Epoch 14 / 50
Average training loss: 0.03 : 100% 39/39 [00:02<00:00, 15.82it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 25.00it/s]
UAS: 72.2295121078373
LAS: 67.78798361763887


Epoch 15 / 50
Average training loss: 0.03 : 100% 39/39 [00:02<00:00, 15.58it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.73it/s]
UAS: 72.68281044971967
LAS: 68.46793113046245


Epoch 16 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 14.94it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.11it/s]
UAS: 73.2593741301841
LAS: 69.00870809972564


Epoch 17 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 15.21it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.68it/s]
UAS: 73.28720823889618
LAS: 69.11606823332936


Epoch 18 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 15.07it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.63it/s]
UAS: 73.78424589446897
LAS: 69.64889260010338


Epoch 19 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 14.95it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.79it/s]
UAS: 73.52578631357112
LAS: 69.3745278142272


Epoch 20 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 14.77it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.25it/s]
UAS: 71.94719471947195
LAS: 67.5573581454531


Epoch 21 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 14.77it/s]
Evaluating validation performance:
100% 2001/2001 [01:18<00:00, 25.37it/s]
UAS: 74.16994711519345
LAS: 70.11411984571951


Epoch 22 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 15.55it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.92it/s]
UAS: 73.8677482206052
LAS: 69.7125134200167


Epoch 23 / 50
Average training loss: 0.03 : 100% 39/39 [00:02<00:00, 16.00it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.23it/s]
UAS: 74.43635929858046
LAS: 70.51572627142232


Epoch 24 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 15.91it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.03it/s]
UAS: 73.83593781064853
LAS: 69.93916259095789


Epoch 25 / 50
Average training loss: 0.03 : 100% 39/39 [00:02<00:00, 15.85it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.31it/s]
UAS: 73.38263946876616
LAS: 69.07630522088354


Epoch 26 / 50
Average training loss: 0.03 : 100% 39/39 [00:02<00:00, 15.99it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.23it/s]
UAS: 74.74651079565788
LAS: 70.72249393614061


Epoch 27 / 50
Average training loss: 0.04 : 100% 39/39 [00:02<00:00, 16.12it/s]
Evaluating validation performance:
100% 2001/2001 [01:18<00:00, 25.51it/s]
UAS: 73.21165851524911
LAS: 69.27909658435723


Epoch 28 / 50
Average training loss: 0.05 : 100% 39/39 [00:02<00:00, 15.87it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.29it/s]
UAS: 69.65684520259255
LAS: 65.65270984929818


Epoch 29 / 50
Average training loss: 0.11 : 100% 39/39 [00:02<00:00, 15.88it/s]
Evaluating validation performance:
100% 2001/2001 [01:18<00:00, 25.43it/s]
UAS: 71.43822816016541
LAS: 67.25913555210943


Epoch 30 / 50
Average training loss: 0.16 : 100% 39/39 [00:02<00:00, 15.86it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.24it/s]
UAS: 73.29516084138534
LAS: 69.34669370551514


Epoch 31 / 50
Average training loss: 0.19 : 100% 39/39 [00:02<00:00, 15.86it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.30it/s]
UAS: 75.00894667780031
LAS: 71.09228995188676


Epoch 32 / 50
Average training loss: 0.27 : 100% 39/39 [00:02<00:00, 16.00it/s]
Evaluating validation performance:
100% 2001/2001 [01:22<00:00, 24.18it/s]
UAS: 75.25547735496441
LAS: 70.96107201081554


Epoch 33 / 50
Average training loss: 0.20 : 100% 39/39 [00:02<00:00, 16.04it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.48it/s]
UAS: 75.98314048272297
LAS: 72.19372539663605


Epoch 34 / 50
Average training loss: 0.06 : 100% 39/39 [00:02<00:00, 15.87it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.60it/s]
UAS: 77.14819674738558
LAS: 73.51783371108195


Epoch 35 / 50
Average training loss: 0.02 : 100% 39/39 [00:02<00:00, 15.97it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.41it/s]
UAS: 77.70885522287169
LAS: 74.22959163386219


Epoch 36 / 50
Average training loss: 0.01 : 100% 39/39 [00:02<00:00, 15.96it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.80it/s]
UAS: 77.78440494651875
LAS: 74.31309395999841


Epoch 37 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.82it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.52it/s]
UAS: 77.89971768261164
LAS: 74.53179052845043


Epoch 38 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.89it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.69it/s]
UAS: 77.86790727265497
LAS: 74.52383792596127


Epoch 39 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.73it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.95it/s]
UAS: 77.94743329754662
LAS: 74.62722175832042


Epoch 40 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.05it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.83it/s]
UAS: 77.9315280925683
LAS: 74.59143504711918


Epoch 41 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.03it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.88it/s]
UAS: 78.00707781621536
LAS: 74.6749373732554


Epoch 42 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 14.84it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.61it/s]
UAS: 77.87983617638872
LAS: 74.57155354089626


Epoch 43 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.98it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 24.98it/s]
UAS: 78.03093562368285
LAS: 74.72662928943497


Epoch 44 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.98it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.55it/s]
UAS: 77.94743329754662
LAS: 74.62722175832042


Epoch 45 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.88it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.02it/s]
UAS: 78.11443794981908
LAS: 74.82603682054952


Epoch 46 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 16.28it/s]
Evaluating validation performance:
100% 2001/2001 [01:20<00:00, 25.00it/s]
UAS: 77.9315280925683
LAS: 74.61529285458667


Epoch 47 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.88it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.12it/s]
UAS: 78.10250904608533
LAS: 74.82603682054952


Epoch 48 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.91it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.69it/s]
UAS: 77.99514891248161
LAS: 74.71072408445664


Epoch 49 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.87it/s]
Evaluating validation performance:
100% 2001/2001 [01:21<00:00, 24.59it/s]
UAS: 78.12239055230825
LAS: 74.88568133921827


Epoch 50 / 50
Average training loss: 0.00 : 100% 39/39 [00:02<00:00, 15.79it/s]
Evaluating validation performance:
100% 2001/2001 [01:19<00:00, 25.11it/s]
UAS: 78.00707781621536
LAS: 74.73458189192414

Evaluation Report after 50 epochs:
UAS: 78.00707781621536
LAS: 74.73458189192414
```

## Results

### Final UAS and LAS


![Result_UAS_LAS](https://user-images.githubusercontent.com/63910248/207279216-09fefb6c-fffe-49aa-ac20-c193f8a374ce.PNG)



