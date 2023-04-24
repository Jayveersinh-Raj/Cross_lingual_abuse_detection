# Cross Lingual Zero Shot Transfer
Cross-lingual NLP: Developing NLP models that can effectively process and translate multiple languages, especially low-resource languages, to help bridge language barriers and make information more accessible.

# Authors:
Jayveersinh Raj

Makar Shevchenko

Nikolay Pavlenko

# Brief Description
This is a project for `Abuse reporting` trained on `toxic comments by Jigsaw Google dataset with 150k+ english comments`. The project aims to accomplish the arbitary zero shot transfer for abuse detection in arbitarary language while being trained on English dataset. It attempts to achieve this by using the vector space alignment that is the core idea behind embedding models like XLM-Roberta, MUSE etc. Different embeddings are tested with the dataset to check the best performing embedder. Our project/model can be used by any platform or software engineer/enthusiast who has to deal with multiple languages to directly flag the toxic behaviour, or identify a valid report by users for a toxic behaviour. The use case for this can be application specific, but the idea is to make the model work with arbitary language by training on a singular language data available.

# The architectural diagram
<img width="634" alt="image" src="https://user-images.githubusercontent.com/69463767/232441899-c594e5cc-762d-4834-bf86-8087287861bc.png">

### NOTE: The classifier architecture can have arbitrary parameters, or hidden states, the above diagram is a general idea. Diagram Credits: Samuel Leonardo Gracio

# Similar work
Daily motion (Credits : Samuel Leonardo Gracio)
![image](https://user-images.githubusercontent.com/69463767/232442675-cf573b1c-c243-4d25-860a-dafa30bb186e.png)

# Dataset Description and link
[jigsaw-toxic-comment-classification](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

We merged all the classes to one, since all the classes belong to one super class of toxicity. Our hypothesis is to use this to flag severe toxic behaviour, severe enough to ban or block a user.