from sklearn.neighbors import NearestNeighbors
import pandas as pd
import logging
from flair.data import Sentence
from flair.embeddings import (
    WordEmbeddings,
    FlairEmbeddings,
    DocumentPoolEmbeddings)

logging.basicConfig(format='%(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

dataset = 'employeur-questions'
data = pd.read_csv("data/{}.csv".format(dataset), sep='\t',
                   encoding="utf-8", engine='python')
data.reset_index(inplace=True)

nrows = data.shape[0]
logging.info("dataset {dataset} is loaded with {nrows} rows".format(
    dataset=dataset, nrows=nrows))

# init Flair embeddings
glove_embedding = WordEmbeddings('fr')
flair_forward_embedding = FlairEmbeddings('multi-forward')
flair_backward_embedding = FlairEmbeddings('multi-backward')

document_embeddings = DocumentPoolEmbeddings([
    glove_embedding,
    flair_forward_embedding,
    flair_backward_embedding
])
logging.info("embedding is loaded")


def embed_content(row):
    index = row["index"]
    if index % 10 == 0:
        logging.info("{} / {}".format(index, nrows))
    sentence = Sentence(row["content"])
    document_embeddings.embed(sentence)
    return list(sentence.get_embedding())


data["embed"] = data.apply(lambda row: embed_content(row), axis=1)
logging.info("data is embedded")

neigh = NearestNeighbors(n_neighbors=2)
neigh.fit(list(data["embed"]))


neighbors = neigh.kneighbors(
    list(data["embed"]))[1]
logging.info("model is fitted")

data["check"], data["nearest"] = [list(i) for i in zip(*neighbors)]
logging.info("neighbors are predicted")


def get_nearest_content(row):
    index = row["index"]
    if index % 10 == 0:
        logging.info("{} / {}".format(index, nrows))
    nearest_index = row["nearest"]
    return data.loc[nearest_index, :]["content"]


data["nearest_content"] = data.apply(
    lambda row: get_nearest_content(row), axis=1)
logging.info("nearest content is found")

data.to_csv("data/{}-processed.csv".format(dataset), sep='\t')
