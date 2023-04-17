from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import utils

def test_tfidf_vectorizer():
    corpus = ["This is the first document.",
              "This is the second second document.",
              "And the third one.",
              "Is this the first document?"]
    expected_output = np.array([[0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
                                    0.        , 0.35872874, 0.        , 0.43877674],
                                [0.        , 0.27230147, 0.        , 0.27230147, 0.        ,
                                    0.85322574, 0.22262429, 0.        , 0.27230147],
                                [0.55280532, 0.        , 0.        , 0.        , 0.55280532,
                                    0.        , 0.28847675, 0.55280532, 0.        ],
                                [0.        , 0.43877674, 0.54197657, 0.43877674, 0.        ,
                                    0.        , 0.35872874, 0.        , 0.43877674]])
    np.testing.assert_array_almost_equal(
        utils.tfidf_vectorizer(corpus).toarray(), expected_output, decimal=6)

def test_cosine_distance():
    vector1 = np.array([1, 2, 3, 4])
    vector2 = np.array([5, 6, 7, 8])
    expected_distance = 0.9688639316269669
    np.testing.assert_almost_equal(utils.cosine_distance(vector1, vector2), expected_distance)