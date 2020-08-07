import re
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    """Generates a dataset for Smith 

    Args:
        file: str, path to raw dataset where each line is a data sample
        tokenizer: BertTokenizer, a tokenizer to generate tokens and convert to ids
        sentence_block_length: Int, Maximum length of a sentence block
        max_sentence_blocks: Int, Maximum number of sentence blocks
        mask: Boolean, Set to True if tokens are to be masked
    """
    
    def __init__(self, file, tokenizer, sentence_block_length, max_sentence_blocks, mask=False):
        """Initialization"""
        self.tokenizer = tokenizer
        self.sentence_block_length = sentence_block_length
        self.num_blocks = max_sentence_blocks

        self.padding_token = tokenizer._parameters['pad_token']
        self.padding_id = tokenizer.token_to_id(self.padding_token)
        self.mask_token = tokenizer._parameters['mask_token']
        self.mask_id = tokenizer.token_to_id(self.mask_token)
        self.sep_token = tokenizer._parameters['sep_token']
        self.sep_token_id = tokenizer.token_to_id(self.sep_token)
        self.mask = mask

        self.mask_prob = 0.15
        self.keep_mask_prob = 0.8
        self.random_mask_prob = 0.1

        self.vocab_size = tokenizer.get_vocab_size()
        self.data = self.read_file(file)
        self.preprocess(self.data)
        
    def read_file(self, file):
        """Returns list of paragaphs"""
        f = open(file, encoding='utf-8')
        data = f.read()
        f.close()
        data = data.split('\n')
        data = [entry for entry in data if entry]

        data = self.split_sentences(data)

        return data
		
    def split_sentences(self, data):
        """Splits the input data into sentences

        Splits based on the following rules for punctuation marks. 
        1. Preceding character is not a number
        2. Preceding character is not a capital number
        3. Following character is a space

        Args:
            data: list, where each entry is a sentence
        Returns:
            data: list, list where each entry is a list of sentences
        """

        data = [re.split('(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', line) for line in data]
		
        return data

    def preprocess(self, data):
        """Preprocesses documents into sentence blocks

        Each document is separated into a sentence block based on the sentence_block_length. 
        The sentence block is filled up with sentences and padded when the length will be 
        greater than the sentence_block_length. Sentences longer than the sentence_block_length 
        are truncated. The number of sentence blocks in each document is stored in doc_split_idx.
        A document will only be included if it is has 3 sentence blocks due to 2 sentence blocks 
        getting masked.		

        Args:
            data: list, each entry is a document of string
        Returns:
            None
        """
        
        doc_token_ids = []
        doc_split_idx = []

        for line in data:
            current_idx = 0
            sentence_idx = 0
            sentence_token_ids = [[] for block in range(self.num_blocks)]

            current_sentence = line[current_idx]
            current_sentence_tokens = self.tokenizer.encode(current_sentence).ids
            current_sentence_tokens = current_sentence_tokens[:-1] # Remove SEP token
            sentence_token_ids[sentence_idx] = current_sentence_tokens

            current_idx += 1

            while current_idx < len(line):

                if sentence_idx > self.num_blocks:
                    break

                current_sentence = line[current_idx]
                current_sentence_tokens = self.tokenizer.encode(current_sentence).ids
                current_sentence_tokens = current_sentence_tokens[1:-1] # Remove CLS and SEP tokens

                num_tokens_so_far = len(sentence_token_ids[sentence_idx])
                num_tokens = len(current_sentence_tokens)

                if num_tokens_so_far + num_tokens <= self.sentence_block_length:
                    sentence_token_ids[sentence_idx].extend(current_sentence_tokens) 
                    current_idx += 1
                elif num_tokens_so_far == 0 and num_tokens >= self.sentence_block_length:
                    sentence_token_ids[sentence_idx] = sentence_token_ids[sentence_idx][0:self.sentence_block_length]
                    sentence_idx += 1
                    current_idx += 1
                else:
                    num_pad_tokens = self.sentence_block_length - num_tokens_so_far
                    sentence_token_ids[sentence_idx].extend([self.padding_id]*num_pad_tokens)
                    sentence_idx += 1
      
            # Pad the last sentence block
            num_pad_tokens = self.sentence_block_length - len(sentence_token_ids[sentence_idx])
            sentence_token_ids[sentence_idx].extend([self.padding_id]*num_pad_tokens)

            # Set final non-padded token to be SEP token
            #sentence_token_ids[sentence_idx][-1] = self.sep_token_id

            # remove empty sentence blocks
            sentence_token_ids = [token_list for token_list in sentence_token_ids if token_list]

            if len(sentence_token_ids) >= 3:
                doc_token_ids.append(sentence_token_ids)
                doc_split_idx.append(len(sentence_token_ids))

        self.doc_token_ids = doc_token_ids
        self.doc_split_idx = doc_split_idx
        self.token_type_ids = torch.zeros(self.sentence_block_length, dtype=torch.long)
        self.position_ids = torch.arange(0, self.sentence_block_length)

    def __len__(self):
        """Denotes the total number of samples"""

        return len(self.doc_token_ids)

    def __getitem__(self, index):
        """Generates one sample of data

        Args:
            index: int, current index of dataset
        Returns:
		    token_ids_sample: tensor, 
            attention_mask: tensor, mask so that pad tokens are not computed during self-attention 
            token_type_ids: tensor, segment token ids to indicate first (0) and second segment (1)
            label_ids: tensor, labels for masked language model. Set to -100 if token is not masked
            split_idx_sample: int, number of sentence blocks in the document
        """

        # Select sample
        token_ids_sample = torch.tensor(self.doc_token_ids[index])
        split_idx_sample = self.doc_split_idx[index]

        attention_mask = torch.ones(size=token_ids_sample.shape, dtype=torch.long)

        # set all tokens that are not [MASK] to -100
        label_ids = torch.ones(token_ids_sample.shape, dtype=torch.long)*-100

        if self.mask:
            # Since the pad token (0) is the smallest token we find it's first 
            # occurence in order to determine the number of non padded tokens 
            # in each sentence. If there is no pad token set index to the 
            # sentence block length
            _, find_first_pad_token = torch.topk(token_ids_sample, k=1, dim=1, 
                                                 largest=False)
            find_first_pad_token[find_first_pad_token == 0] = self.sentence_block_length
			
            num_tokens_to_mask = torch.ceil(self.mask_prob*find_first_pad_token).long()

            # insert mask tokens on the fly 80% receive mask token, 10%  random token, 10% keep token
            for i in range(token_ids_sample.shape[0]):
                mask_ids = torch.randint(high=find_first_pad_token[i].item(), 
                                         size=(num_tokens_to_mask[i].item(),))

                random_insert = torch.rand(num_tokens_to_mask[i].item())

                label_ids[i, mask_ids] = token_ids_sample[i][mask_ids] 

                
                num_tokens_replaced_from_vocab = torch.sum(random_insert >= (1 - self.random_mask_prob)).item()
                
                token_ids_sample[i, mask_ids[random_insert >= (1 - self.random_mask_prob)]] = \
                    torch.randint(high=self.vocab_size, 
                                  size=(num_tokens_replaced_from_vocab,))

                token_ids_sample[i, mask_ids[random_insert < self.keep_mask_prob]] = self.mask_id

            attention_mask[token_ids_sample == 0] = 0

        return (token_ids_sample, attention_mask, self.token_type_ids, label_ids, split_idx_sample)