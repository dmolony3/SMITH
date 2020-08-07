def collate_fun(batch):
    """Generate a batch in list format"""

    token_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    token_type_ids = [item[2] for item in batch]
    label_ids = [item[3] for item in batch]
    split_ids = [item[4] for item in batch]

    return [token_ids, attention_mask, token_type_ids, label_ids, split_ids]