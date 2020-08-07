# SMITH - Siamese multi-depth transformer based hierarchical encoder

This repository is a pytorch implementation of [SMITH](https://arxiv.org/abs/2004.12297). SMITH is a transformer model for learning document representations. It consists of a hierarchy of 2 BERT transformer models. The first transformer encodes blocks of sentences while the second transformer takes the *[CLS]* output of a documents encoded sentence blocks and outputs a document representation. Self-attention is performed over words in sentence blocks and then over the sentence blocks.

Two encoded documents can be compared by computing the cosine similarity of their encoded documents. Similar documents will have a high cosine similarity while dissimilar ones will have a low value.

### **WORK IN PROGRESS**

## Usage
### Installation
pip install -r requirements.txt

### Data Requirements
Documents should be stored in a txt/csv file where each line corresponds to a document. 


### Pretraining
To pretrain the model from the command line run the folowing. 


```
python main.py --file_path=/home/documents.csv
```

The number of layers, attention heads for each transformer as well as the sentence block length can be set as follows
```
python main.py  --file_path=/home/documents.csv --num_layers1=6, --num_layesr2=4 --heads1=8 --heads2=4 --block_length=64
```

A full list of optional arguments can be found from the command line
```
python main.py --help
```
