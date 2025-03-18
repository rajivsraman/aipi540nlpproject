from gensim.models import Word2Vec

def train_word2vec(texts):
    model = Word2Vec(texts, vector_size=100, min_count=1, window=5, workers=3, epochs=10, sg=0, negative=5) 
    return model