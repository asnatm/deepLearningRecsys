# Item embedding for session based recommendation
This section includes the materials for the session based recommendation use case.
It is based on our paper "Item Embedding for Session Based Recommendation" https://drive.google.com/open?id=1RIdOKa6DEAW0MdWFcj4JX3pgrcUyigJI

## Installation requirements 
- Python version 3.6 (anaconda installation)  https://www.anaconda.com/download/
- Tensorflow (backend) 
- Keras version 2.1 https://keras.io/
- GloVe (optional) http://nlp.stanford.edu/projects/glove/
- Recsys 15' challenge dataset http://2015.recsyschallenge.com/challenge.html (a shorter version is available in this repository)

## Workshop structure
The hands-on exercise includes the following steps:
- Pre-processing 1: train\test, corpus file to GloVe
- GloVe (optional): vocab, vectors files
- Pre-processing 2 (optional):  NN in - sequence format + word embedding, NN out – one hot by vocab
- Model train
- Model test (calc recall@10)

## Materials
- data : a short version of the recsys 15' challenge data
- pre-train : outputs for each step to save time during the workshop
- solution : jupyter notebook solution
- root: presentation, python code, jupyter notebook for each step (besides GloVe)


