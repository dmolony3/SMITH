import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
import transformers
from tokenizers import BertWordPieceTokenizer
from transformers import BertConfig, BertModel, BertForMaskedLM
from data_generator import Dataset
from data_utils import collate_fun

# SMITH parameters (batch size 32, Adam - learning rate 5e-5, beta1 0.9, beta2 - 0.999, epsilon=1e-6, dropout=0.1)
# Sentence level - 6 layers, 4 attention heads, hidden size - 256, sentence block length - 32, mask 2 sentence blocks per document
# Document level - 3 layers, 4 attention heads, hidden size - 256, document length - 1024 (by setting max number of sentence blocks to 32)

file_path = r'C:\Users\David\Documents\Machine_learning\NLP\CardioExplorer\abstracts.csv'

tokenizer = BertWordPieceTokenizer(r'C:\Users\David\Documents\Machine_learning\NLP\CardioExplorer\vocab.txt', lowercase=True)

pretrain = True
sentence_block_length = 32
max_sentence_blocks = 48
hidden_size = 256
batch_size = 4
shuffle = True
drop_last = True

sentence_block_vector = torch.normal(mean=0.0, std=1.0, size=[hidden_size])

sentence_config = BertConfig()
sentence_config.vocab_size = tokenizer.get_vocab_size()
sentence_config.num_hidden_layers = 6
sentence_config.hidden_size = 256
sentence_config.num_attention_heads = 4
sentence_config.max_position_embeddings = sentence_block_length # sentence_block_length

document_config = BertConfig()
document_config.vocab_size = tokenizer.get_vocab_size()
document_config.num_hidden_layers = 3
document_config.hidden_size = 256
document_config.num_attention_heads = 4
document_config.max_position_embeddings = max_sentence_blocks # sentence_block_length

dataset=Dataset(file_path, tokenizer, sentence_block_length, 
    max_sentence_blocks, mask=True)
dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last, collate_fn=collate_fun)


#sentence_model = AutoModel.from_config(sentence_config)
#document_model = AutoModel.from_config(document_config)
sentence_model = BertForMaskedLM(sentence_config)
document_model = BertModel(document_config)

dense1 = torch.nn.Linear(sentence_config.hidden_size, sentence_config.hidden_size)
dense2 = torch.nn.Linear(document_config.hidden_size, document_config.hidden_size)


for iteration, (token_ids, attention_mask, token_type_ids, label_ids, split_idx) in enumerate(dataloader):
    token_ids_stacked = torch.cat(token_ids)
    label_ids_stacked = torch.cat(label_ids)
    attention_mask = torch.cat(attention_mask)
    token_type_ids = torch.cat(token_type_ids)
	
    output = sentence_model(token_ids_stacked, attention_mask, lm_labels=label_ids_stacked)

    loss_wp = output[0]

    cls_sentence = output[1][:, 0, :]
    cls_sentence = dense1(cls_sentence)
    cls_sentence_norm = f.normalize(cls_sentence, dim=1 ,p=2)
	
    # resplit batch into invididual documents
    cls_sentence_norm_split = torch.split(cls_sentence_norm, split_idx, dim=0)

    if pretrain:
        # fill each document with zeros so that all documents have the same shape
        num_sentence_blocks = max(split_idx)

        doc_input_list = []
        for i in range(batch_size):
            num_rows_to_pad = num_sentence_blocks - split_idx[i]
            zero_padding =  torch.zeros(num_rows_to_pad, sentence_config.hidden_size)
            doc_input_list.append(torch.cat([cls_sentence_norm_split[i], zero_padding], dim=0))
		
        sentence_block_embeddings, sentence_block_labels, mask_indices = mask_sentence_blocks(doc_input_list, num_to_mask, split_idx)
        attention_mask = torch.ones_like(sentence_block_embeddings)
        attention_mask[sentence_block_embeddings==0] = 0
    else:
        sentence_block_embeddings = cls_sentence_norm_split

	
    output = document_model(inputs_embeds=document_input_embeddings, attention_mask=attention_mask)

    # gather at each masked sentence
    if pretrain:
        # project word model to embedding space
        prediction_scores = project_word_model(output[0])
        document_output_embeddings = output[0].view(-1, document_config.hidden_size) # check accuracy of this
        masked_output_embeddings = document_output_embeddings[mask_indices, :]
        
        loss_sp = masked_sentence_block_loss(masked_output_embeddings, sentence_block_labels)

        loss = loss_sp + loss_wp
        if iteration % 10 == 0:
            print("Iteration {}: Loss: {}".format(iteration, loss)) 
    else:
        cls_document = output[0][:, 0, :]
        cls_document = dense2(cls_document)
        cls_document_norm = f.normalize(cls_document, dim=1 ,p=2)

        doc_embedding = combine_output(cls_document_norm, cls_sentence_norm_split, concat_mode="sum")


def masked_word_model_loss():
    """Computes the masked word language modeling loss"""
    loss_fct = CrossEntropyLoss()
    loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
	
    return loss

def masked_sentence_block_loss(predicted_embeddings, label_embeddings):
    """Computes the masked sentence block loss"""

    sim = torch.tensordot(predicted_embeddings.unsqueeze(1), label_embeddings.T.unsqueeze(0), dims=2)

    B = sim.shape[0]
	
    mask = torch.eye(B).long()

    p = torch.exp(sim)/torch.sum(torch.exp(sim), 1)

    log_likelihood = torch.log(p)*mask

    negative_log_likelihood = -torch.sum(log_likelihood)/B

    return negative_log_likelihood
	
    """
    l = torch.zeros(B, B)
    l = []

    for i in range(B):
        for j in range(B):
            if j == i:
                l.append(torch.exp(sim[j, i])/torch.sum(torch.exp(sim[j, :])))
    l = sum(l)/B

    for i in range(B):
        for j in range(B):
                l[i, j] = torch.exp(sim[j, i])/torch.sum(torch.exp(sim[j, :]))
    l = -torch.sum(l*torch.eye(B).long())/B
    """

def mask_sentence_blocks(sentence_block_batch, num_to_mask, split_idx):
    """Randomly masks sentence block vectors"""
    #sentence_block_ground_truth = []
    sentence_block_labels = []
    mask_indices = []
    num_sentence_blocks = max(split_idx)
    for i in range(len(sentence_block_batch)):
        mask_idx = torch.randperm(split_idx[i])[:num_to_mask]
        #sentence_block_ground_truth.append(sentence_block_batch[i][mask_idx, :]) # use either ground truth or labels approach
        sentence_block_labels.append(sentence_block_batch[i][mask_idx, :])
        sentence_block_batch[i][mask_idx, :] = sentence_block_vector
        mask_indices.extend(mask_idx + num_sentence_blocks*i)
		
    sentence_block_labels = torch.cat(sentence_block_labels)
    sentence_block_batch = torch.cat(sentence_block_batch)
    return sentence_block_batch, sentence_block_labels, mask_indices
    

def combine_output(doc_output, sentence_output, concat_mode="sum"):
    """Combines document and sentence representations

    Args:
        doc_output: tensor, batch of normalized document representations
        sentence_output: list, each entry is a tensor of sentence block representations
        concat_mode: string, method used to combine sentence and document representations
    Returns:
        output:
    """

    if concat_mode == "sum":
        sentence_output = [torch.sum(sentence_embedding, axis=0) for sentence_embedding in sentence_output]
        sentence_output = torch.stack(sentence_output)
        output = torch.cat([doc_output, sentence_output], axis=1)
    elif concat_mode == "mean":
        sentence_output = [torch.mean(sentence_embedding, axis=0) for sentence_embedding in sentence_output]
        sentence_output = torch.stack(sentence_output)
        output = torch.cat([doc_output, sentence_output], axis=1)
    elif concat_mode == "attention":
        projection_mat = torch.nn.Linear(sentence_config.hidden_size, V)
        output = torch.cat([doc_output, sentence_output], axis=1)    
    else:
        output = doc_output

    return output
