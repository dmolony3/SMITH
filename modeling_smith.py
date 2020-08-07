import torch
import torch.nn.functional as f
from loss import masked_sentence_block_loss
from mask_sentences import mask_sentence_blocks
from transformers import BertConfig, BertModel, BertForMaskedLM

class SmithModel(torch.nn.Module):
    """Siamese multi-depth transformer based hierarchical encoder

    Smith consists of 2 transformer models. The first transformer models
    accepts sentence blocks as input and computes the masked language 
    mask loss. The second transformer takes the normalized output from the
    sentence level transformer and computes a document level representation.
    The input to the document level transformer is masked and the masked
    sentence block loss is computed against the ground truth sentence 
    block representations. Both losses are returned.

    Args:
        sentence_config:
        document_config:
    Returns:
        None:

    """
    def __init__(self, sentence_config, document_config):
        super().__init__()
        
        self.sentence_hidden_size = sentence_config.hidden_size
        self.document_hidden_size = document_config.hidden_size

        self.sentence_model = BertForMaskedLM(sentence_config)
        self.document_model = BertModel(document_config)

        self.dense1 = torch.nn.Linear(sentence_config.hidden_size, sentence_config.hidden_size)
        self.dense2 = torch.nn.Linear(document_config.hidden_size, document_config.hidden_size)

        self.sentence_block_mask_vector = torch.normal(mean=0.0, std=1.0, size=[self.document_hidden_size])
        self.num_masked_blocks = document_config.num_masked_blocks

    def forward(self, 
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        split_idx=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        # output[0] = loss
        # output[1] = scores for token predictions
        # output[2] = list of hidden states for embeddings and each layer
        sentence_output = self.sentence_model(input_ids, attention_mask, masked_lm_labels=labels, output_hidden_states=True)

        loss_wp = sentence_output[0]

        cls_sentence = sentence_output[2][-1][:, 0, :]
        cls_sentence = self.dense1(cls_sentence)
        cls_sentence_norm = f.normalize(cls_sentence, dim=1 ,p=2)

        # resplit batch into invididual documents
        cls_sentence_norm_split = torch.split(cls_sentence_norm, split_idx, dim=0)
	
        num_sentence_blocks = max(split_idx)

        doc_input_list = []

        # pad each document with zeros so that all documents have the same shape
        for i in range(len(split_idx)):
            num_rows_to_pad = num_sentence_blocks - split_idx[i]
            zero_padding =  torch.zeros(num_rows_to_pad, self.sentence_hidden_size)
            doc_input_list.append(torch.cat([cls_sentence_norm_split[i], zero_padding], dim=0))
		
        sentence_block_embeddings, sentence_block_labels, mask_indices = mask_sentence_blocks(doc_input_list, self.num_masked_blocks, split_idx, self.sentence_block_mask_vector)

        attention_mask = torch.ones_like(sentence_block_embeddings)
        attention_mask[sentence_block_embeddings==0] = 0
        attention_mask = torch.sum(attention_mask, dim=2)
        attention_mask[attention_mask > 0] = 1

        doc_output = self.document_model(inputs_embeds=sentence_block_embeddings, attention_mask=attention_mask)

        document_output_embeddings = doc_output[0].view(-1, self.document_hidden_size) # check accuracy of this
        masked_output_embeddings = document_output_embeddings[mask_indices, :]
        
        cls_document = doc_output[0][:, 0, :]
        cls_document = self.dense2(cls_document)
        cls_document_norm = f.normalize(cls_document, dim=1 ,p=2)

        loss_sp = masked_sentence_block_loss(masked_output_embeddings, sentence_block_labels)

        outputs = (loss_sp, loss_wp,) + (doc_output[0],)+ (cls_sentence_norm_split,)+ (cls_document_norm,)

        return outputs