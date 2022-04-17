# Deep Learning for Healthcare
## RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance

This repo is based on the original RefDNN publication and repo (https://github.com/mathcom/RefDNN).  
Details can be found in the publication: *Choi, Jonghwan, et al. "RefDNN: a reference drug based neural network for more accurate prediction of anticancer drug resistance." Scientific Reports 10 (2020):1-11* (https://www.nature.com/articles/s41598-020-58821-x).

**Overview of the RefRNN architecture:**  

![RefRNN_architecture](./assets/RefRNN_architecture.png)


The RefDNN project aims to predict cancer drug resistance and proposed a “reference drug” based neural network architecture. The task is to build a classifier to predict whether a cancer cell line is sensitive or resistant to a certain drug. The key idea of this work is to use a set of drugs (so-called reference drugs) to learn the representations of cell lines based on gene expression and drugs by their molecular structure.



--------------------------------------------------------------------------------------------
## SYSTEM REQUIERMENTS: 

   - RefDNN requires system memory larger than 24GB.
    
   - If you want to use tensorflow-gpu, GPU memory of more than 4GB is required.


--------------------------------------------------------------------------------------------
## PYTHON ENVIRONMENT SETUP:

Note that the original code was based on an older version of tensorflow. Here the code has been updated using `tf_upgrade_v2`. However, tensorflow 1.15 is recommended here because potential errors might arise using tensorflow 2.xx. Python 3.7 is reccommended to run tensorflow 1.15. Setup of virtual environment for running tensorflow-gpu is as follows:  

```
conda create --name tf python=3.7
conda init bash
# (restart the shell)
conda activate tf
conda install -c conda-forge tensorflow-gpu=1.15
pip install numpy pandas scikit-learn scikit-optimize scipy jupyter

# test if gpu is available:
python
import tensorflow as tf
tf.test.is_gpu_available() # True
```


--------------------------------------------------------------------------------------------
## DATA:

All data can be found in the original repo. There are two pharmacogenomics datasets used in this study: the Cancer Cell Line Encyclopaedia (CCLE) and Genomics of Drug Sensitivity in Cancer (GDSC) datasets. Each dataset has three components:  


|                                           | GDSC                                              | CCLE                                         |
|-------------------------------------------|---------------------------------------------------|----------------------------------------------|
| Gene expression of cell lines (predictor) | 983 cell lines x 17780 genes                      | 491 cell lines x 18926 genes                 |
| Fingerprint of drugs (predictor)          | 222 drugs x 3072 dimensions                       | 12 drugs x 3072 dimensions                   |
| Drug response (response)                  | 190036 pairs (120606 resistance; 69430 sensitive) | 5724 pairs (3402 resistance; 2322 sensitive) |


--------------------------------------------------------------------------------------------
## RefDNN MODEL:

The model can be found in `refdnn.py` (based on the RefRNN repo) or the `RefRNN_model_notebook.ipynb` notebook.  

    
--------------------------------------------------------------------------------------------
## MODEL TRAINING:

The `RefDNN_training_notebook.ipynb` notebook is an interactive model training interface to run the nested cross validation on the CCLE dataset as an example. The script is based on the original script from the RefRNN repo.  
    

--------------------------------------------------------------------------------------------
