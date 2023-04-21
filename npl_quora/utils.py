from __future__ import annotations
import scipy
import pickle
import errno
import os
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics
import gensim.models.word2vec as w2v
import tqdm
import spacy

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
        vector1: scipy.sparse.csr.csr_matrix | np.ndarray, 
        vector2: scipy.sparse.csr.csr_matrix | np.ndarray) -> float:
    if isinstance(vector1, scipy.sparse.csr.csr_matrix):
        return np.dot(
            vector1.T.toarray()[0]/np.linalg.norm(vector1.toarray()),
            vector2.T.toarray()[0]/np.linalg.norm(vector2.toarray()))
    else:
        return np.dot(
            vector1/np.linalg.norm(vector1),
            vector2/np.linalg.norm(vector2))
          

def same_words_ordered(q1_tokens,q2_tokens):
    n = min(len(q1_tokens), len(q2_tokens))
    same = 0
    for i in range(n):
        if q1_tokens[i] == q2_tokens[i]:
            same += 1

    return same / n

def generate_ordered_words_feature(q1_tokens,q2_tokens):
    ow_feature = []
    for i in range(len(q1_tokens)):
        ow = same_words_ordered(q1_tokens[i],q2_tokens[i])
        ow_feature.append(ow)
    return ow_feature


def search_key_word(q_tokens):
    for token in q_tokens:
        if token in ['where', 'when', 'who', 'do', 'should']:
            return token
        elif token in ['can', 'how', 'could']:
            return 'can'
        elif token in ['what', 'which']:
            return 'what'
        elif token in ['why', 'whey']:
            return 'why'
        else: return None


def compare_key_word(q1_tokens, q2_tokens):
    key_word1 = search_key_word(q1_tokens)
    key_word2 = search_key_word(q2_tokens)
    if key_word1 and key_word1 == key_word2:
        return 1
    else:
        return 0
    

def generate_key_words_feature(q1_tokens, q2_tokens):
    kw_feature = []
    for i in range(len(q1_tokens)):
        kw = compare_key_word(q1_tokens[i],q2_tokens[i])
        kw_feature.append(kw)
    return kw_feature


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
        tokens:list[list[str]],
        n_features:int,
        seed:int = 1,
        workers = 1,
        sg:int = 0,
        context_size:int = 5,
        down_sampling:int = 1e-3,
        min_word_count:int = 0) -> w2v:

    return w2v.Word2Vec(
        sentences=tokens,
        sg=sg,
        seed=seed,
        workers = workers,
        vector_size = n_features,
        min_count = min_word_count,
        window = context_size,
        sample = down_sampling
    )

def w2v_embedding(
        tokens: list[list[str]], 
        wv: w2v.wv) -> np.ndarray:
    
    sentence_vectors = []
    for sentence in tokens:
        word_vectors = []
        for token in sentence:
            word_vectors.append(wv.get_vector(token))
        sentence_vectors.append(list(np.mean(word_vectors, axis=0)))

    return np.array(sentence_vectors)

def calculate_metrics(y,X,model):
    model_metrics = []
    model_metrics.append(metrics.roc_auc_score(y,model.predict(X)))
    model_metrics.append(metrics.accuracy_score(y,model.predict(X)))
    model_metrics.append(metrics.precision_score(y,model.predict(X)))
    model_metrics.append(metrics.recall_score(y,model.predict(X)))
    model_metrics.append(metrics.f1_score(y,model.predict(X)))
    return model_metrics

def save_model(model, filename):
    if not os.path.exists('model_artifacts'):
        try:
            # create folder
            os.makedirs('model_artifacts')
            # open file descriptors for writing
            file_path = open(f'model_artifacts/{filename}.pkl', 'wb')
            pickle.dump(model, file_path)
            # close file descriptors
            file_path.close()
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    else:
        # open file descriptors for writing
        file_path = open(f'model_artifacts/{filename}.pkl', 'wb')
        pickle.dump(model, file_path)
        # close file descriptors
        file_path.close()