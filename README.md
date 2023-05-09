# Cross-Lingual Zero-Shot Transfer Learning for Toxic Comments Detection
In this work we applied multilingual zero-shot transfer concept for the task of toxic comments detection. This concept allows a model trained only on a single-language dataset to work in arbitrary language, even low-resource. We achieved the concept by using embedding models XLM_RoBERTa and DistilBERT that transform a text from any language to a single vector space. We demonstrate that a classifier trained on "toxic comments by Jigsaw Google" English dataset can reach 75% accuracy on a manually created multilingual dataset of 50 different languages. The applications of our models include flagging toxic comments in mutlilingual social platforms. We share all the code and data for training and deployment in our repository in GitHub. 

Authors:
- Jayveersinh Raj
- Makar Shevchenko
- Nikolay Pavlenko

Report on the project can be accessed [here](https://github.com/SyrexMinus/cross_lingual_nlp/blob/main/progress_reports/project_technical_report.pdf).

## Pipeline

The figure below illustrates sample model pipeline. The pipeline consist of an embedder followed by a classifier model. In fact, in place of the classifier in the project, neural network, naive bayes and decision tree were tested.

<img width="634" alt="image" src="https://user-images.githubusercontent.com/69463767/232441899-c594e5cc-762d-4834-bf86-8087287861bc.png">

Credit: Samuel Leonardo Gracio

## Motivation

The idea of implementing a zero-shot multilingual model is to cover rare languages without the need for additional training in them. The figure below illustrates the distribution of languages by video in some video streaming service. This illustrates that minority languages are used much less often than English or French. Accordingly, there is much less data for them, which creates a problem for training models in such languages. However, using the zero-shot technique allows inference in such rare languages without using additional training data.

![Daily motion](https://user-images.githubusercontent.com/69463767/232442675-cf573b1c-c243-4d25-860a-dafa30bb186e.png)

Credit: Samuel Leonardo Gracio

## Dataset

The dataset that we use, namely jigsaw-toxic-comment-classification was taken from Kaggle. It could be accessed through [this](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) link.

In preprocessing step we merged all the classes of toxicity to one super-class to deal with sparsity of them. The expected application of our model (ban toxic comments) allows not to distinguish specifics of toxicity.

## Tech stack

In the work we used the following tools and frameworks:

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
