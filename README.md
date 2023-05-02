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

# Tech stack
<a href="https://pytorch.org/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/pytorch/pytorch-icon.svg" alt="pytorch" width="50" height="50"/> </a>
<a href="https://www.python.org" target="_blank" rel="noreferrer"> <img src="https://media3.giphy.com/media/LMt9638dO8dftAjtco/200.webp?cid=ecf05e473jsalgnr0edawythfdeh3o2gnrisk725vn7x9n72&rid=200.webp&ct=s" alt="python" width="50" height="50"/> </a> 
<a href="https://huggingface.co/" target="_blank" rel="noreferrer"> <img src="https://media3.giphy.com/media/BGLSkombEDjGEJ41oW/giphy.webp?cid=ecf05e47fu5099qknyuij1yq6exe2eylr2pv3y4toyqlk535&ep=v1_stickers_search&rid=giphy.webp&ct=s" alt="python" width="50" height="50"/> </a> 
<a href="https://jupyter.org/" target="_blank" rel="noreferrer"> <img alt="Jupyter Notebook" width="50" height="50" src="https://img.icons8.com/fluency/344/jupyter.png"></a>
<a href="https://numpy.org/doc/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/numpy/numpy-icon.svg" alt="NumPy" width="50" height="50"/> </a>
<a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://media0.giphy.com/media/p7l6subf8WlFK/200.webp?cid=ecf05e472j8ufhiqbsz74tfghvw67xyg4skm5z8ejqldvg6f&rid=200.webp&ct=s" alt="pandas" width="50" height="50"/> </a>
<a href="https://matplotlib.org/stable/index.html" target="_blank" rel="noreferrer"> <img src="https://seeklogo.com/images/M/matplotlib-logo-AEB3DC9BB4-seeklogo.com.png" alt="Matplotlib" width="60" height="40"/> </a>
<a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="50" height="50"/> </a>
 <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> <img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="50" height="50"/> </a>
<a href="https://streamlit.io/" target="_blank" rel="noreferrer"> <img src="https://user-images.githubusercontent.com/69463767/235664976-da8d40b1-9332-48f9-a73f-bd62c7060b32.png" alt="seaborn" width="50" height="40"/> </a>
<a href="https://onnx.ai/" target="_blank" rel="noreferrer"> <img src="https://www.vectorlogo.zone/logos/onnxai/onnxai-icon.svg" alt="seaborn" width="50" height="50"/> </a>
<a href="https://developer.nvidia.com/cuda-toolkit" target="_blank" rel="noreferrer"> <img src="https://www.svgrepo.com/show/373541/cuda.svg" alt="seaborn" width="50" height="50"/> </a>
<a href="https://developer.nvidia.com/tensorrt" target="_blank" rel="noreferrer"> <img src="https://user-images.githubusercontent.com/69463767/235667402-0584035a-8ce6-4d6b-ae66-66c8ff6c084c.png" alt="seaborn" width="80" height="50"/> </a>

