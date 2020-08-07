from transformers import BertConfig

def set_model_config(args, tokenizer):
    sentence_config = BertConfig()
    sentence_config.vocab_size = tokenizer.get_vocab_size()
    sentence_config.num_hidden_layers = args.num_layers1
    sentence_config.hidden_size = args.hidden_size1
    sentence_config.num_attention_heads = args.attention_heads1
    sentence_config.max_position_embeddings = args.block_length 

    document_config = BertConfig()
    document_config.vocab_size = tokenizer.get_vocab_size()
    document_config.num_hidden_layers = args.num_layers2
    document_config.hidden_size = args.hidden_size2
    document_config.num_attention_heads = args.attention_heads2
    document_config.num_masked_blocks = args.max_blocks
    document_config.max_position_embeddings = args.max_blocks

    return sentence_config, document_config