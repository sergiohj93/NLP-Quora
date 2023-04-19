import scipy
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.models.word2vec as w2v
import multiprocessing
import spacy
from tqdm import tqdm

def cast_list_as_strings(mylist):
    """
    return a list of strings
    """
    #assert isinstance(mylist, list), f"the input mylist should be a list it is {type(mylist)}"
    mylist_of_strings = []
    for x in mylist:
        mylist_of_strings.append(str(x))

    return mylist_of_strings


def get_features_from_df(df, count_vectorizer):
    """
    returns a sparse matrix containing the features build by the count vectorizer.
    Each row should contain features from question1 and question2.
    """
    q1_casted =  cast_list_as_strings(list(df["question1"]))
    q2_casted =  cast_list_as_strings(list(df["question2"]))
    
    ############### Begin exercise ###################
    # what is kaggle                  q1
    # What is the kaggle platform     q2
    X_q1 = count_vectorizer.transform(q1_casted)
    X_q2 = count_vectorizer.transform(q2_casted)

    X_q1q2 = scipy.sparse.hstack((X_q1,X_q2))
    ############### End exercise ###################

    return X_q1q2


def get_mistakes(clf, X_q1q2, y):

    ############### Begin exercise ###################
    predictions = clf.predict(X_q1q2)
    incorrect_predictions = predictions != y 
    incorrect_indices,  = np.where(incorrect_predictions)
    
    ############### End exercise ###################
    
    if np.sum(incorrect_predictions)==0:
        print("no mistakes in this df")
    else:
        return incorrect_indices, predictions
    

def print_mistake_k(k, train_df, mistake_indices, predictions):
    print(train_df.iloc[mistake_indices[k]].question1)
    print(train_df.iloc[mistake_indices[k]].question2)
    print("true class:", train_df.iloc[mistake_indices[k]].is_duplicate)
    print("prediction:", predictions[mistake_indices[k]])
    
    
def lower_list(mylist):
    list_lowered = []
    for string in mylist:
        list_lowered.append(string.lower())
    return list_lowered

def remove_sw(mylist,stop_words):
    list_without_sw = []
    for string in mylist:
        # Pattern to match stop words
        pattern = re.compile(r'\b(' + '|'.join(stop_words) + r')\b', re.IGNORECASE)
        # Remove the stop words using the regular expression pattern
        list_without_sw.append(pattern.sub('',string))
    return list_without_sw
    
def tokenize(mylist):
    list_tokenized = []
    for string in mylist:
        # Regular expression pattern to match words
        pattern = re.compile(r"\w+")
        tokens = pattern.findall(string)
        if len(tokens)==0:
            tokens=[""]
        list_tokenized.append(tokens)
    return list_tokenized

def jaccard_similarity(sent1,sent2):
    return len(sent1.intersection(sent2)) / len(sent1.union(sent2))
    
def jaccard_distance(sent1,sent2):
    return 1-jaccard_similarity(sent1,sent2)

def generate_jd_feature(q1_tokens,q2_tokens):
    jd_feature = []
    for i in range(len(q1_tokens)):
        jd = jaccard_distance(set(q1_tokens[i]),set(q2_tokens[i]))
        jd_feature.append(jd)
    return jd_feature
        
def tfidf_vectorizer(
        corpus: list[str],
        max_features=1000) -> TfidfVectorizer:
    vectorizer = TfidfVectorizer(
        max_features=max_features)
    return vectorizer.fit(corpus)

def cosine_distance(
        vector1: np.array, 
        vector2: np.array) -> float:
    return np.dot(
        vector1.T/np.linalg.norm(vector1),
        vector2.T/np.linalg.norm(vector2))            

def same_word(token1, token2):
    if token1 == token2:
        return 1
    
    return 0

def generate_first_word_feature(q1_tokens, q2_tokens):
    fw_feature = []
    for i in range(len(q1_tokens)):
        fw = same_word(q1_tokens[i][0],q2_tokens[i][0])
        fw_feature.append(fw)
    return fw_feature

# No sé si es muy útil (con checkear la primera palabra y el average word embedding creo que es mejor).
def same_words_ordered(q1_tokens,q2_tokens):
    n = min(len(q1_tokens), len(q2_tokens))
    same = 0
    for i in range(n):
        same += same_word(q1_tokens[i], q2_tokens[i])

    return same / n

def generate_ordered_words_feature(q1_tokens,q2_tokens):
    ow_feature = []
    for i in range(len(q1_tokens)):
        ow = same_words_ordered(q1_tokens[i],q2_tokens[i])
        ow_feature.append(ow)
    return ow_feature

def negation(sent1, sent2, nlp):
    doc1 = nlp(sent1)
    doc2 = nlp(sent2)

    # Check if the negation is the same
    if (not any(token.dep_ == 'neg' for token in doc1)) == (not any(token.dep_ == 'neg' for token in doc2)):
        return 1
    else:
        return 0
    
def generate_negation_feature(q1, q2):
    nlp = spacy.load('en_core_web_sm')

    negation_feature = []
    for i in tqdm(range(len(q1))):
        neg = negation(q1[i],q2[i],nlp)
        negation_feature.append(neg)
    return negation_feature
    
def build_w2v_model(
        doc:list[str],
        n_fueatures:int,
        seed:int = 1,
        sg:int = 0,
        context_size:int = 5,
        down_sampling:int = 1e-3,
        min_word_count:int = 0) -> w2v:
    num_workers = multiprocessing.cpu_count()

    return w2v.Word2Vec(
        sentences=doc,
        sg=sg,
        seed=seed,
        workers = num_workers,
        vector_size = n_fueatures,
        min_count = min_word_count,
        window = context_size,
        sample = down_sampling
    )

def w2v_embedding(
        tokens: list[list[str]], 
        word2vec: w2v) -> np.array:
    
    word_vectors = []
    for sentence in tokens:
        for token in sentence:
            word_vectors.append(word2vec.wv.get_vector(token))

    return np.mean(word_vectors, axis=0)