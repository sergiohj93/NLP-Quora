import scipy
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer

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
        # Regular expression pattern to match stop words
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
        corpus: list[str]) -> np.array:
    vectorizer = TfidfVectorizer()
    return vectorizer.fit_transform(corpus)

def cosine_distance(
        vector1: np.array, 
        vector2: np.array) -> float:
    return np.dot(
        vector1.T/np.linalg.norm(vector1),
        vector2.T/np.linalg.norm(vector2))      
