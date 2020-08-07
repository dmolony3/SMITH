# SMITH - Siamese multi-depth transformer based hierarchical encoder

This repository is a pytorch implementation of [SMITH](https://arxiv.org/abs/2004.12297). SMITH is a transformer model for learning document representations. It consists of a hierarchy of 2 BERT transformer models. The first transformer encodes sentence block while the second transformer takes the [CLS] output of the encoded sentence blocks for a documents and outputs a document representation.

# **WORK IN PROGRESS**

## Usage
### Installation
pip install -r requirements.txT

### Data Requirements
Documents should be stored in a txt/csv file where each line corresponds to a document. 


### Pretraining
To pretrain the model from the command line run the folowing. 


```
python main.py --file_path=/home/documents.csv
```

A list of optional arguments can be found from the command line
```
python main.py --help
```
