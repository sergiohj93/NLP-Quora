from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.models.word2vec as w2v
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

def test_tokenize():
    assert utils.tokenize(["Hello, world!"]) == [["Hello", "world"]]

def test_build_w2v_model():
    # Given
    doc = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
    n_features = 10
    n_epochs = 10

    # When
    model = utils.build_w2v_model(doc, n_features, n_epochs)

    # Then
    assert isinstance(model, w2v.Word2Vec)
    assert model.vector_size == n_features

def test_w2v_embedding():
    # Given
    n_features = 10
    n_epochs = 10
    doc = 'This is a test'

    # When
    embedding = utils.w2v_embedding(n_features, n_epochs, [doc])

    # Then
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (n_features,)