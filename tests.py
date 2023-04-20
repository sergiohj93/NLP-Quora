from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.models.word2vec as w2v
import numpy as np
import utils

def test_tfidf_vectorizer():
    # Given
    doc = ["This is a test", "This is another test"]

    # When
    vectorizer = utils.tfidf_vectorizer(doc)

    # Then
    assert isinstance(vectorizer, TfidfVectorizer)
    assert (vectorizer.get_feature_names_out() == ["another", "is", "test", "this"]).all()

def test_cosine_distance():
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    assert utils.cosine_distance(a, b) == 0.0

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
    doc = [['this', 'is', 'a', 'test'], ['this', 'is', 'another', 'test']]
    n_features = 10
    n_epochs = 10
    model = utils.build_w2v_model(doc, n_features, n_epochs)

    # When
    embedding = utils.w2v_embedding(doc, model)

    # Then
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (2, n_features)