import random
import numpy as np
import tensorflow as tf

from pickle import load, dump
from sys import argv
from tqdm import tqdm
from math import sqrt
from collections import Counter
from typing import List, Dict, Union, Tuple, Any, NamedTuple
from tensorflow.keras import models, layers, optimizers

UNKNOWN = "UNK"
ROOT = "ROOT"
NULL = "NULL"
NONEXIST = -1

class DependencyTree:
    """
    Main class to maintain the state of dependency tree
    and operate on it.
    """

    def __init__(self) -> None:
        self.n = 0
        self.head = [NONEXIST]
        self.label = [UNKNOWN]
        self.counter = -1

    def add(self, head: str, label: str) -> None:
        """
        Add the next token to the parse.
        h: Head of the next token
        l: Dependency relation label between this node and its head
        """
        self.n += 1
        self.head.append(head)
        self.label.append(label)

    def set(self, k, h, l):
        """
        Establish a labeled dependency relation between the two given nodes.
        k: Index of the dependent node
        h: Index of the head node
        l: Label of the dependency relation
        """
        self.head[k] = h
        self.label[k] = l

    def get_head(self, k) -> int:
        if k <= 0 or k > self.n:
            return NONEXIST
        return self.head[k]

    def get_label(self, k) -> int:
        if k <= 0 or k > self.n:
            return NULL
        return self.label[k]

    def get_root(self) -> int:
        """
        Get the index of the node which is the root of the parse
        (i.e., the node which has the ROOT node as its head).
        """
        for k in range(1, self.n+1):
            if self.get_head(k) == 0:
                return k
        return 0

    def is_single_root(self) -> bool:
        """
        Check if this parse has only one root.
        """
        roots = 0
        for k in range(1, self.n+1):
            if self.get_head(k) == 0:
                roots += 1
        return roots == 1

    def is_tree(self) -> bool:
        """
        Check if the tree is legal.
        """
        h = []
        h.append(-1)
        for i in range(1, self.n+1):
            if self.get_head(i) < 0 or self.get_head(i) > self.n:
                return False
            h.append(-1)

        for i in range(1, self.n+1):
            k = i
            while k > 0:
                if h[k] >= 0 and h[k] < i:
                    break
                if h[k] == i:
                    return False
                h[k] = i
                k = self.get_head(k)

        return True

    """
    Check if the tree is projective
    """
    def is_projective(self) -> bool:
        if not self.is_tree():
            return False
        self.counter = -1
        return self.visit_tree(0)

    def visit_tree(self, w) -> bool:
        """
        Inner recursive function for checking projective of tree
        """
        for i in range(1, w):
            if self.get_head(i) == w and not self.visit_tree(i):
                return False
        self.counter += 1
        if w != self.counter:
            return False
        for i in range(w+1, self.n+1):
            if self.get_head(i) == w and not self.visit_tree(i):
                return False
        return True

    def equal(self, t) -> bool:
        if t.n != self.n:
            return False
        for i in range(1, self.n+1):
            if self.get_head(i) != t.get_head(i):
                return False
            if self.get_label(i) != t.get_label(i):
                return False
        return True

    def print_tree(self) -> None:
        for i in range(1, self.n+1):
             print(str(i) + " " + str(self.get_head(i)) + " " + self.get_label(i))
        print("\n")

class Configuration:

    def __init__(self, sentence):
        self.stack = []
        self.buffer = []
        self.tree = DependencyTree()
        self.sentence = sentence

    def shift(self):
        k = self.get_buffer(0)
        if k == NONEXIST:
            return False
        self.buffer.pop(0)
        self.stack.append(k)
        return True

    def remove_second_top_stack(self):
        n_stack = self.get_stack_size()
        if n_stack < 2:
            return False
        self.stack.pop(-2)
        return True

    def remove_top_stack(self):
        n_stack = self.get_stack_size()
        if n_stack <= 1:
            return False
        self.stack.pop()
        return True

    def get_stack_size(self):
        return len(self.stack)

    def get_buffer_size(self):
        return len(self.buffer)

    def getSentenceSize(self):
        return len(self.sentence)

    def get_head(self, k):
        return self.tree.get_head(k)

    def get_label(self, k):
        return self.tree.get_label(k)

    def get_stack(self, k):
        """
            Get the token index of the kth word on the stack.
            If stack doesn't have an element at this index, return NONEXIST
        """
        n_stack = self.get_stack_size()
        if k >= 0 and k < n_stack:
            return self.stack[n_stack-1-k]
        return NONEXIST

    def get_buffer(self, k):
        """
        Get the token index of the kth word on the buffer.
        If buffer doesn't have an element at this index, return NONEXIST
        """
        if k >= 0 and k < self.get_buffer_size():
            return self.buffer[k]
        return NONEXIST

    def get_word(self, k):
        """
        Get the word at index k
        """
        if k == 0:
            return ROOT
        else:
            k -= 1

        if k < 0 or k >= len(self.sentence):
            return NULL
        return self.sentence[k].word

    def get_pos(self, k):
        """
        Get the pos at index k
        """
        if k == 0:
            return ROOT
        else:
            k -= 1

        if k < 0 or k >= len(self.sentence):
            return NULL
        return self.sentence[k].pos

    def add_arc(self, h, t, l):
        """
        Add an arc with the label l from the head node h to the dependent node t.
        """
        self.tree.set(t, h, l)

    def get_left_child(self, k, cnt):
        """
            Get cnt-th leftmost child of k.
            (i.e., if cnt = 1, the leftmost child of k will be returned,
                   if cnt = 2, the 2nd leftmost child of k will be returned.)
        """
        if k < 0 or k > self.tree.n:
            return NONEXIST

        c = 0
        for i in range(1, k):
            if self.tree.get_head(i) == k:
                c += 1
                if c == cnt:
                    return i
        return NONEXIST

    def get_right_child(self, k, cnt):
        """
        Get cnt-th rightmost child of k.
        (i.e., if cnt = 1, the rightmost child of k will be returned,
               if cnt = 2, the 2nd rightmost child of k will be returned.)
        """
        if k < 0 or k > self.tree.n:
            return NONEXIST

        c = 0
        for i in range(self.tree.n, k, -1):
            if self.tree.get_head(i) == k:
                c += 1
                if c == cnt:
                    return i
        return NONEXIST

    def has_other_child(self, k, goldTree):
        for i in range(1, self.tree.n+1):
            if goldTree.get_head(i) == k and self.tree.get_head(i) != k:
                return True
        return False

    def get_str(self):
        """
            Returns a string that concatenates all elements on the stack and buffer, and head / label
        """
        s = "[S]"
        for i in range(self.get_stack_size()):
            if i > 0:
                s += ","
            s += self.stack[i]

        s += "[B]"
        for i in range(self.get_buffer_size()):
            if i > 0:
                s += ","
            s += self.buffer[i]

        s += "[H]"
        for i in range(1, self.tree.n+1):
            if i > 1:
                s += ","
            s += self.get_head(i) + "(" + self.get_label(i) + ")"

        return s

class ParsingSystem:

    """
    Main class to maintain the state of parsing system
    and operate on it.
    """

    def __init__(self, labels: List[str]) -> None:
        self.single_root = True
        self.labels = labels
        self.transitions = []
        self.root_label = labels[0]
        self.make_transitions()

    def make_transitions(self) -> None:
        """
        Generate all possible transitions which this parsing system can
        take for any given configuration.
        """
        for label in self.labels:
            self.transitions.append("L(" + label + ")")
        for label in self.labels:
            self.transitions.append("R(" + label + ")")

        self.transitions.append("S")

    def initial_configuration(self, sentence) -> Configuration:
        configuration = Configuration(sentence)
        length = len(sentence)

        # For each token, add dummy elements to the configuration's tree
        # and add the words onto the buffer
        for i in range(1, length+1):
            configuration.tree.add(NONEXIST, UNKNOWN)
            configuration.buffer.append(i)

        # Put the ROOT node on the stack
        configuration.stack.append(0)

        return configuration

    def is_terminal(self, configuration: Configuration) -> bool:
        return configuration.get_stack_size() == 1 and configuration.get_buffer_size() == 0

    def get_oracle(self,
                   configuration: Configuration,
                   tree: DependencyTree) -> str:
        """
        Provide a static-oracle recommendation for the next parsing step to take
        """
        word1 = configuration.get_stack(1)
        word2 = configuration.get_stack(0)
        if word1 > 0 and tree.get_head(word1) == word2:
            return "L(" + tree.get_label(word1) + ")"
        elif word1 >= 0 and tree.get_head(word2) == word1 and not configuration.has_other_child(word2, tree):
            return "R(" + tree.get_label(word2) + ")"
        return "S"

    def can_apply(self, configuration: Configuration, transition: str) -> bool:
        """
        Determine whether the given transition is legal for this
        configuration.
        """
        if transition.startswith("L") or transition.startswith("R"):
            label = transition[2:-1]
            if transition.startswith("L"):
                h = configuration.get_stack(0)
            else:
                h = configuration.get_stack(1)
            if h < 0:
                return False
            if h == 0 and label != self.root_label:
                return False

        n_stack = configuration.get_stack_size()
        n_buffer = configuration.get_buffer_size()

        if transition.startswith("L"):
            return n_stack > 2
        elif transition.startswith("R"):
            if self.single_root:
                return (n_stack > 2) or (n_stack == 2 and n_buffer == 0)
            else:
                return n_stack >= 2
        return n_buffer > 0

    def apply(self, configuration: Configuration, transition: str) -> Configuration:

        """
        =================================================================

        Implement arc standard algorithm based on
        Incrementality in Deterministic Dependency Parsing(Nirve, 2004):
        Left-reduce
        Right-reduce
        Shift

        =================================================================
        """

        if transition.startswith("L"):
            word1, word2, label = configuration.get_stack(1), configuration.get_stack(0), transition[2:-1]
            configuration.add_arc(word2, word1, label)
            configuration.remove_second_top_stack()
        elif transition.startswith("R"):
            word1, word2, label = configuration.get_stack(1), configuration.get_stack(0), transition[2:-1]
            configuration.add_arc(word1, word2, label)
            configuration.remove_top_stack()
        else:
            configuration.shift()

        return configuration

    def num_transitions(self) -> int:
        return len(self.transitions)

    def print_transitions(self) -> None:
        for transition in self.transitions:
            print(transition)

    def get_punctuation_tags(self) -> List[str]:
        return ["''", ",", ".", ":", "``", "-LRB-", "-RRB-"]

    def evaluate(self, sentences, trees, gold_trees) -> str:
        """
        Evaluate performance on a list of sentences, predicted parses, and gold parses
        """
        result = []
        punctuation_tags = self.get_punctuation_tags()

        if len(trees) != len(gold_trees):
            print("Incorrect number of trees.")
            return None

        correct_arcs = 0
        correct_arcs_no_punc = 0
        correct_heads = 0
        correct_heads_no_punc = 0

        correct_trees = 0
        correct_trees_no_punc = 0
        correct_root = 0

        sum_arcs = 0
        sum_arcs_no_punc = 0

        for i in range(len(trees)):
            tree = trees[i]
            gold_tree = gold_trees[i]
            tokens = sentences[i]

            if tree.n != gold_tree.n:
                print("Tree", i+1, ": incorrect number of nodes.")
                return None

            if not tree.is_tree():
                print("Tree", i+1, ": illegal.")
                return None

            n_correct_head = 0
            n_correct_head_no_punc = 0
            n_no_punc = 0

            for j in range(1, tree.n+1):
                if tree.get_head(j) == gold_tree.get_head(j):
                    correct_heads += 1
                    n_correct_head += 1
                    if tree.get_label(j) == gold_tree.get_label(j):
                        correct_arcs += 1
                sum_arcs += 1

                tag = tokens[j-1].pos
                if tag not in punctuation_tags:
                    sum_arcs_no_punc += 1
                    n_no_punc += 1
                    if tree.get_head(j) == gold_tree.get_head(j):
                        correct_heads_no_punc += 1
                        n_correct_head_no_punc += 1
                        if tree.get_label(j) == gold_tree.get_label(j):
                            correct_arcs_no_punc += 1

            if n_correct_head == tree.n:
                correct_trees += 1
            if n_correct_head_no_punc == n_no_punc:
                correct_trees_no_punc += 1
            if tree.get_root() == gold_tree.get_root():
                correct_root += 1

        result = ""
        result += "UAS: " + str(correct_heads * 100.0 / sum_arcs) + "\n"
        #result += "UASnoPunc: " + str(correct_heads_no_punc * 100.0 / sum_arcs_no_punc) + "\n"
        result += "LAS: " + str(correct_arcs * 100.0 / sum_arcs) + "\n"
        #result += "LASnoPunc: " + str(correct_arcs_no_punc * 100.0 / sum_arcs_no_punc) + "\n\n"

        #result += "UEM: " + str(correct_trees * 100.0 / len(trees)) + "\n"
        #result += "UEMnoPunc: " + str(correct_trees_no_punc * 100.0 / len(trees)) + "\n"
        #result += "ROOT: " + str(correct_root * 100.0 / len(trees)) + "\n"

        return result

class Vocabulary:

    def __init__(self,
                 sentences,
                 trees) -> None:
        self.word_token_to_id = {}
        self.pos_token_to_id = {}
        self.label_token_to_id = {}
        self.id_to_token = {}

        word = []
        pos = []
        label = []
        for sentence in sentences:
            for token in sentence:
                word.append(token.word)
                pos.append(token.pos)

        root_label = None
        for tree in trees:
            for k in range(1, tree.n + 1):
                if tree.get_head(k) == 0:
                    root_label = tree.get_label(k)
                else:
                    label.append(tree.get_label(k))

        if root_label in label:
            label.remove(root_label)

        index = 0
        word_count = [UNKNOWN, NULL, ROOT]
        word_count.extend(Counter(word))
        for word in word_count:
            self.word_token_to_id[word] = index
            self.id_to_token[index] = word
            index += 1

        pos_count = [UNKNOWN, NULL, ROOT]
        pos_count.extend(Counter(pos))
        for pos in pos_count:
            self.pos_token_to_id[pos] = index
            self.id_to_token[index] = pos
            index += 1

        label_count = [NULL, root_label]
        label_count.extend(Counter(label))
        for label in label_count:
            self.label_token_to_id[label] = index
            self.id_to_token[index] = label
            index += 1

    def get_word_id(self, token: str) -> int:
        if token in self.word_token_to_id:
            return self.word_token_to_id[token]
        return self.word_token_to_id[UNKNOWN]

    def get_pos_id(self, token: str):
        if token in self.pos_token_to_id:
            return self.pos_token_to_id[token]
        return self.pos_token_to_id[UNKNOWN]

    def get_label_id(self, token: str):
        if token in self.label_token_to_id:
            return self.label_token_to_id[token]
        return self.label_token_to_id[UNKNOWN]

    def save(self, pickle_file_path: str) -> None:
        with open(pickle_file_path, "wb") as file:
            dump(self, file)

    @classmethod
    def load(cls, pickle_file_path: str) -> 'Vocabulary':
        with open(pickle_file_path, "rb") as file:
            vocabulary = load(file)
        return vocabulary

class Token(NamedTuple):

    word: str = None
    pos: str = None
    head: int = None
    dep_type: str = None

Sentence = List[Token]

def read_conll_data(data_file_path: str) -> Tuple[List[Sentence], List[DependencyTree]]:

    """
    Reads Sentences and Trees from a CONLL formatted data file.
    """

    sentences: List[Sentence] = []
    trees: List[DependencyTree] = []

    with open(data_file_path, 'r') as file:
        sentence_tokens = []
        tree = DependencyTree()
        for line in tqdm(file):
            line = line.strip()
            array = line.split('\t')
            if len(array) < 10:
                if sentence_tokens:
                    trees.append(tree)
                    sentences.append(sentence_tokens)
                    tree = DependencyTree()
                    sentence_tokens = []
            else:
                word = array[1]
                pos = array[4]
                head = int(array[6])
                dep_type = array[7]
                token = Token(word=word, pos=pos,
                              head=head, dep_type=dep_type)
                sentence_tokens.append(token)
                tree.add(head, dep_type)

    if not sentences:
        raise Exception(f"No sentences read from {data_file_path}. ")

    return sentences, trees


def write_conll_data(output_file: str,
                     sentences: List[Sentence],
                     trees: List[DependencyTree]) -> None:
    """
    Writes Sentences and Trees into a CONLL formatted data file.
    """
    with open(output_file, 'w') as fout:
        for i in range(len(sentences)):
            sent = sentences[i]
            tree = trees[i]
            for j in range(len(sent)):
                fout.write("%d\t%s\t_\t%s\t%s\t_\t%d\t%s\t_\t_\n"
                           % (j+1, sent[j].word, sent[j].pos,
                              sent[j].pos, tree.get_head(j+1), tree.get_label(j+1)))
            fout.write("\n")

def generate_training_instances(parsing_system: ParsingSystem,
                                sentences: List[List[str]],
                                vocabulary: Vocabulary,
                                trees: List[DependencyTree]) -> List[Dict]:
    """
    Generates training instances of configuration and transition labels
    from the sentences and the corresponding dependency trees.
    """
    num_transitions = parsing_system.num_transitions()
    instances: Dict[str, List] = []
    for i in tqdm(range(len(sentences))):
        if trees[i].is_projective():
            c = parsing_system.initial_configuration(sentences[i])
            while not parsing_system.is_terminal(c):
                oracle = parsing_system.get_oracle(c, trees[i])
                feature = get_configuration_features(c, vocabulary)
                label = []
                for j in range(num_transitions):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.can_apply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)
                if 1.0 not in label:
                    print(i, label)
                instances.append({"input": feature, "label": label})
                c = parsing_system.apply(c, oracle)
    return instances

def get_configuration_features(configuration: Configuration,
                               vocabulary: Vocabulary) -> List[List[int]]:

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """

    features = list()
    fWord = []
    fPos = []
    fLabel = []
    feature = []

    for j in range(2, -1, -1):
        index = configuration.get_stack(j)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))


    for j in range(0, 3, 1):
        index = configuration.get_buffer(j)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))

    for j in range(0, 2, 1):
        k = configuration.get_stack(j)
        index = configuration.get_left_child(k, 1)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))
        fLabel.append(vocabulary.get_label_id(configuration.get_label(index)))

        index = configuration.get_right_child(k, 1)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))
        fLabel.append(vocabulary.get_label_id(configuration.get_label(index)))

        index = configuration.get_left_child(k, 2)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))
        fLabel.append(vocabulary.get_label_id(configuration.get_label(index)))

        index = configuration.get_right_child(k, 2)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))
        fLabel.append(vocabulary.get_label_id(configuration.get_label(index)))

        index = configuration.get_left_child(configuration.get_left_child(k, 1), 1)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))
        fLabel.append(vocabulary.get_label_id(configuration.get_label(index)))

        index = configuration.get_right_child(configuration.get_right_child(k, 1), 1)
        fWord.append(vocabulary.get_word_id(configuration.get_word(index)))
        fPos.append(vocabulary.get_pos_id(configuration.get_pos(index)))
        fLabel.append(vocabulary.get_label_id(configuration.get_label(index)))


    features.extend(fWord)
    features.extend(fPos)
    features.extend(fLabel)

    assert len(features) == 48
    return features

def generate_batches(instances: List[Dict],
                     batch_size: int) -> List[Dict[str, np.ndarray]]:

    """
    Generates and returns batch of tensorized instances in a chunk of batch_size.
    """

    def chunk(items: List[Any], num: int) -> List[Any]:
        return [items[index:index+num] for index in range(0, len(items), num)]
    batches_of_instances = chunk(instances, batch_size)

    batches = []
    for batch_of_instances in tqdm(batches_of_instances):
        count = min(batch_size, len(batch_of_instances))
        features_count = len(batch_of_instances[0]["input"])

        batch = {"inputs": np.zeros((count, features_count), dtype=np.int32)}
        if "label" in  batch_of_instances[0]:
            labels_count = len(batch_of_instances[0]["label"])
            batch["labels"] = np.zeros((count, labels_count), dtype=np.int32)

        for batch_index, instance in enumerate(batch_of_instances):
            batch["inputs"][batch_index] = np.array(instance["input"])
            if "label" in instance:
                batch["labels"][batch_index] = np.array(instance["label"])

        batches.append(batch)

    return batches

def load_embeddings(embeddings_txt_file: str,
                    vocabulary: Vocabulary,
                    embedding_dim: int) -> np.ndarray:

    vocab_id_to_token = vocabulary.id_to_token
    tokens_to_keep = set(vocab_id_to_token.values())
    vocab_size = len(vocab_id_to_token)

    embeddings: Dict[str, np.ndarray] = {}
    print("\nReading pretrained embedding file.")
    with open(embeddings_txt_file) as file:
        for line in tqdm(file):
            line = str(line).strip()
            token = line.split(' ', 1)[0]
            if not token in tokens_to_keep:
                continue
            fields = line.rstrip().split(' ')
            if len(fields) - 1 != embedding_dim:
                raise Exception(f"Pretrained embedding vector and expected "
                                f"embedding_dim do not match for {token}.")
            vector = np.asarray(fields[1:], dtype='float32')
            embeddings[token] = vector

    embedding_matrix = np.random.normal(size=(vocab_size, embedding_dim),
                                        scale=1./sqrt(embedding_dim))
    embedding_matrix = np.asarray(embedding_matrix, dtype='float32')

    for idx, token in vocab_id_to_token.items():
        if token in embeddings:
            embedding_matrix[idx] = embeddings[token]

    return embedding_matrix

class CubicActivation(layers.Layer):
    def call(self, vector: tf.Tensor) -> tf.Tensor:
        return tf.pow(vector, 3)

class DependencyParser(models.Model):
    def __init__(self,
                 embedding_dim: int,
                 vocab_size: int,
                 num_tokens: int,
                 hidden_dim: int,
                 num_transitions: int,
                 regularization_lambda: float,
                 trainable_embeddings: bool,
                 activation_name: str = "cubic") -> None:
        """
        This model defines a transition-based dependency parser which makes
        use of a classifier powered by a neural network. The neural network
        accepts distributed representation inputs: dense, continuous
        representations of words, their part of speech tags, and the labels
        which connect words in a partial dependency parse.

        This is an implementation of the method described in

        Danqi Chen and Christopher Manning.
        A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

        Parameters
        ----------
        embedding_dim : ``str``
            Dimension of word embeddings
        vocab_size : ``int``
            Number of words in the vocabulary.
        num_tokens : ``int``
            Number of tokens (words/pos) to be used for features
            for this configuration.
        hidden_dim : ``int``
            Hidden dimension of feedforward network
        num_transitions : ``int``
            Number of transitions to choose from.
        regularization_lambda : ``float``
            Regularization loss fraction lambda as given in paper.
        trainable_embeddings : `bool`
            Is the embedding matrix trainable or not.
        """
        super(DependencyParser, self).__init__()
        self._regularization_lambda = regularization_lambda

        if activation_name == "cubic":
            self._activation = CubicActivation()
        elif activation_name == "sigmoid":
            self._activation = tf.keras.activations.sigmoid
        elif activation_name == "tanh":
            self._activation = tf.keras.activations.tanh
        else:
            raise Exception(f"activation_name: {activation_name} is from the known list.")

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_tokens = num_tokens
        self.num_transitions = num_transitions

        self.embeddings = tf.Variable(
            tf.random.uniform(
                [self.vocab_size, self.embedding_dim],
                minval=-0.01, maxval=0.01, dtype=tf.float32
            ),
            trainable=trainable_embeddings
        )

        # weight_hidden = hidden_dim, num_tokens*embedding_dim
        self.weight_hidden = tf.Variable(
            tf.compat.v2.random.normal(
                [self.num_tokens * self.embedding_dim, self.hidden_dim],
                mean=0,
                stddev=1.0 / sqrt(self.num_transitions)
            ),
            trainable=trainable_embeddings
        )

        self.weight_output = tf.Variable(
            tf.compat.v2.random.normal(
                [self.hidden_dim, self.num_transitions, ],
                mean=0,
                stddev=1.0 / sqrt(self.num_transitions)
            ),
            trainable=trainable_embeddings
        )

        self.bias = tf.Variable(
            tf.zeros([self.hidden_dim, 1]),
            trainable=trainable_embeddings
        )

    def call(self,
             inputs: tf.Tensor,
             labels: tf.Tensor = None) -> Dict[str, tf.Tensor]:

        """
        Forward pass of Dependency Parser.

        Parameters
        ----------
        inputs : ``tf.Tensor``
            Tensorized version of the batched input text. It is of shape:
            (batch_size, num_tokens) and entries are indices of tokens
            in to the vocabulary. These tokens can be word or pos tag.
            Each row corresponds to input features a configuration.
        labels : ``tf.Tensor``
            Tensor of shape (batch_size, num_transitions)
            Each row corresponds to the correct transition that
            should be made in the given configuration.

        Returns
        -------
        An output dictionary consisting of:
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """

        embeddings_inputs = tf.nn.embedding_lookup(self.embeddings, inputs)
        embeddings_inputs = tf.reshape(
            embeddings_inputs, [
                embeddings_inputs.shape[0],
                embeddings_inputs.shape[1] * embeddings_inputs.shape[2]
            ]
        )

        hidden_layer = tf.add(
            tf.matmul(embeddings_inputs, self.weight_hidden),
            tf.transpose(self.bias)
        )

        logits = self._activation(hidden_layer)
        logits = tf.matmul(logits, self.weight_output)

        output_dict = {"logits": logits}

        if labels is not None:
            output_dict["loss"] = self.compute_loss(logits, labels)
        return output_dict


    def compute_loss(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.float32:

        """
        Parameters
        ----------
        logits : ``tf.Tensor``
            A tensor of shape ``(batch_size, num_transitions)`` representing
            logits (unnormalized scores) for the labels for every instance in batch.

        Returns
        -------
        loss : ``tf.float32``
            If input has ``labels``, then mean loss for the batch should
            be computed and set to ``loss`` key in the output dictionary.

        """

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=tf.argmax(tf.transpose(labels))
        )

        regularised_loss = tf.nn.l2_loss(self.weight_hidden) + \
            tf.nn.l2_loss(self.weight_output) + \
            tf.nn.l2_loss(self.bias) + \
            tf.nn.l2_loss(self.embeddings)

        regularization = (self._regularization_lambda) * regularised_loss

        return tf.reduce_mean(tf.add(loss, regularization))

def predict(model: models.Model,
            sentences: List[Sentence],
            parsing_system: ParsingSystem,
            vocabulary: Vocabulary) -> List[DependencyTree]:

    """
    Predicts the dependency tree for a given sentence by greedy decoding.
    We generate a initial configuration (features) for ``sentence`` using
    ``parsing_system`` and ``vocabulary``. Then we apply the ``model`` to predict
    what's the best transition for this configuration and apply this transition
    (greedily) with ``parsing_system`` to get the next configuration. We do
    this till the terminal configuration is reached.
    """

    predicted_trees = []
    num_transitions = parsing_system.num_transitions()
    for sentence in tqdm(sentences):
        configuration = parsing_system.initial_configuration(sentence)
        while not parsing_system.is_terminal(configuration):
            features = get_configuration_features(configuration, vocabulary)
            features = np.array(features).reshape((1, -1))
            logits = model(features)["logits"].numpy()
            opt_score = -float('inf')
            opt_trans = ""
            for j in range(num_transitions):
                if (logits[0, j] > opt_score and
                        parsing_system.can_apply(configuration, parsing_system.transitions[j])):
                    opt_score = logits[0, j]
                    opt_trans = parsing_system.transitions[j]
            configuration = parsing_system.apply(configuration, opt_trans)
        predicted_trees.append(configuration.tree)
    return predicted_trees

def Neural_Transition_Dependency_Parser(model: models.Model,
          optimizer: optimizers.Optimizer,
          train_instances: List[Dict[str, np.ndarray]],
          validation_sentences: List[List[str]],
          validation_trees: List[DependencyTree],
          parsing_system: ParsingSystem,
          vocabulary: Vocabulary,
          num_epochs: int,
          batch_size: int) -> Dict[str, Union[models.Model, str]]:
    """
    Trains a model on the given training instances as
    configured and returns the trained model.
    """
    print("\nGenerating Training batches:")
    train_batches = generate_batches(train_instances, batch_size)
    train_batches = [(batch["inputs"], batch["labels"]) for batch in train_batches]

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1} / {num_epochs}")
        # Training Epoch
        total_training_loss = 0
        generator_tqdm = tqdm(train_batches)
        for index, (batch_inputs, batch_labels) in enumerate(generator_tqdm):
            with tf.GradientTape() as tape:
                model_outputs = model(inputs=batch_inputs, labels=batch_labels)
                loss_value = model_outputs["loss"]
                grads = tape.gradient(loss_value, model.trainable_variables)

            clipped_grads = [tf.clip_by_norm(grad, 5) for grad in grads]
            optimizer.apply_gradients(zip(clipped_grads, model.trainable_variables))
            total_training_loss += loss_value
            description = ("Average training loss: %.2f " % (total_training_loss/(index+1)))
            generator_tqdm.set_description(description, refresh=False)

        # Validation evaluation
        print("Evaluating validation performance:")
        predicted_trees = predict(model, validation_sentences, parsing_system, vocabulary)

        """
        Predict the dependency trees and evaluate them comparing with gold trees.
        """
        evaluation_report = parsing_system.evaluate(validation_sentences, predicted_trees, validation_trees)
        print(evaluation_report)

    training_outputs = {"model": model, "evaluation_report": evaluation_report}
    return training_outputs

if __name__ == '__main__':

    batch_size = 10000
    num_epochs = 10
    num_epochs = int(argv[1])

    # Set numpy, tensorflow and python seeds for reproducibility.
    tf.random.set_seed(1337)
    np.random.seed(1337)
    random.seed(13370)

    # Setup Training / Validation data
    print("Reading training data")

    train_data_file_path = "en_ewt-ud-train.conll"

    train_sentences, train_trees = read_conll_data(train_data_file_path)

    print("Reading validation data")

    validation_data_file_path = "en_ewt-ud-dev.conll"

    validation_sentences, validation_trees = read_conll_data(validation_data_file_path)

    vocabulary = Vocabulary(train_sentences, train_trees)

    sorted_labels = [item[0] for item in sorted(vocabulary.label_token_to_id.items(), key=lambda e: e[1])]
    non_null_sorted_labels = sorted_labels[1:]

    parsing_system = ParsingSystem(non_null_sorted_labels)

    print("Generating training instances")
    train_instances = generate_training_instances(parsing_system,
                                                      train_sentences,
                                                      vocabulary, train_trees)

    embedding_dim = 300
    num_tokens = 48
    hidden_dim = 200
    activation_name = "cubic"
    trainable_embeddings = True
    regularization_lambda = 1e-8

    # Setup Model
    config_dict = {"vocab_size": len(vocabulary.id_to_token),
                   "embedding_dim": embedding_dim,
                   "num_tokens": num_tokens,
                   "hidden_dim": hidden_dim,
                   "num_transitions": parsing_system.num_transitions(),
                   "regularization_lambda": regularization_lambda,
                   "trainable_embeddings": trainable_embeddings,
                   "activation_name": activation_name}

    model = DependencyParser(**config_dict)

    pretrained_embedding_file = "glove.6B.300d.txt"

    embedding_matrix = load_embeddings(pretrained_embedding_file, vocabulary, embedding_dim)
    model.embeddings.assign(embedding_matrix)

    # Setup Optimizer
    optimizer = optimizers.Adam()

    # Train
    output = Neural_Transition_Dependency_Parser(model, optimizer, train_instances,
                             validation_sentences, validation_trees,
                             parsing_system, vocabulary, num_epochs,
                             batch_size)

    print("Evaluation Report after", num_epochs, "epochs:")
    print(output["evaluation_report"])
