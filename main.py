import argparse
import torch
import torch.nn.functional as f
from config import set_model_config
from data_generator import Dataset
from data_utils import collate_fun
from modeling_smith import SmithModel
from torch import AdamW, WarmupLinearSchedule
from torch.utils.data import DataLoader
from tokenizers import BertWordPieceTokenizer

# SMITH parameters from paper (batch size 32, Adam - learning rate 5e-5, beta1 0.9, beta2 - 0.999, epsilon=1e-6, dropout=0.1)
# Sentence level - 6 layers, 4 attention heads, hidden size - 256, sentence block length - 32, mask 2 sentence blocks per document
# Document level - 3 layers, 4 attention heads, hidden size - 256, document length - 1024 (by setting max number of sentence blocks to 32)

parser = argparse.ArgumentParser(description='Smith model for learned document representation.')
parser.add_argument('--mode', type=str, default='pretrain', help='Model mode, must be either pretrain or finetune')
parser.add_argument('--file_path', type=str, default=None, help='The path to the training data - a text file where each line is a document')
parser.add_argument('--batch_size', type=int, default=16, help='The number of documents processed during each iteration')
parser.add_argument('--num_epochs', type=int, default=100, help='The number of epochs to train the model for')
parser.add_argument('--num_layers1', type=int, default=6, help='The number of hidden layers in sentence block model')
parser.add_argument('--num_layers2', type=int, default=3, help='The number of hidden layers in document model')
parser.add_argument('--attention_heads1', type=int, default=4, help='The number of attention heads in sentence block model')
parser.add_argument('--attention_heads2', type=int, default=4, help='The number of attention heads in document model')
parser.add_argument('--hidden_size1', type=int, default=256, help='The dimension of the hidden layers in sentence block model')
parser.add_argument('--hidden_size2', type=int, default=256, help='The dimension of the hidden layers in document model')
parser.add_argument('--block_length', type=int, default=32, help='The length each sentence block a document is divided into')
parser.add_argument('--max_blocks', type=int, default=48, help='The maximum number of sentence block per document')

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=t_total) 


def train():

    tokenizer = BertWordPieceTokenizer(r'C:\Users\David\Documents\Machine_learning\NLP\CardioExplorer\vocab.txt', lowercase=True)

    args = parser.parse_args()
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    file_path = args.file_path
	
    file_path = r'C:\Users\David\Documents\Machine_learning\NLP\CardioExplorer\abstracts_100.csv'

    if file_path is None:
        ValueError("A file path to documents must be provided")
    
    sentence_config, document_config = set_model_config(args, tokenizer)

    dataset=Dataset(file_path, tokenizer, sentence_config.max_position_embeddings, 
        document_config.max_position_embeddings, mask=True)

    dataloader = DataLoader(dataset, batch_size=batch_size,
        shuffle=True, drop_last=True, collate_fn=collate_fun)


    model = SmithModel(sentence_config, document_config)

    for epoch in range(num_epochs):
        for iteration, (token_ids, attention_mask, token_type_ids, label_ids, split_idx) in enumerate(dataloader):
            token_ids_stacked = torch.cat(token_ids)
            label_ids_stacked = torch.cat(label_ids)
            attention_mask = torch.cat(attention_mask)
            token_type_ids = torch.cat(token_type_ids)

            output = model(input_ids=token_ids_stacked,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                split_idx=split_idx,
                labels=label_ids_stacked)

            loss_sp = output[0]
            loss_wp = output[1]
            loss = loss_sp + loss_wp

            if iteration % 10 == 0:
                print("Iteration {}: Loss: {}".format(iteration, loss)) 

            loss.backward()

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()


if __name__ == '__main__':
    train()