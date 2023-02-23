import pickle
import numpy as np

model_loc="Applied-Machine-Learning/Assignments/models/lgr.pkl"
lgr=pickle.load(open(model_loc,"rb"))
from sentence_transformers import SentenceTransformer

#Pre-trained model to evaluate the embeddings of the text data
embed_model = SentenceTransformer('all-mpnet-base-v2')


def score(text,model,threshold):
    embedded_data=embed_model.encode([text])
    pred=model.predict(embedded_data)
    propensity=model.predict_proba(embedded_data)[:,1]
    return pred[0],propensity[0]

