import torch

def mask_sentence_blocks(sentence_block_batch, num_masked_blocks, split_idx, sentence_block_mask_vector):
    """Randomly masks sentence block vectors

    Args:
        sentence_block_batch: list, each entry is a tensor of sentence block embeddings
        num_masked_blocks: int, number of blocks to mask in each document
        split_idx: list, each entry indicates the number of sentence blocks in an individual document
        sentence_block_mask_vector: tensor, randomly initialized vector to represent masked sentence blocks
    Returns:
        sentence_block_batch: tensor, batch of sentence blocks where num_masked_blocks have been masked for each document
        sentence_block_labels: tensor, true value for the masked sentence blocks
        mask_indices: list, each entry is the index of a masked sentence block
    """

    sentence_block_labels = []
    mask_indices = []
    num_sentence_blocks = max(split_idx)

    for i in range(len(sentence_block_batch)):
        mask_idx = torch.randperm(split_idx[i])[:num_masked_blocks]
        sentence_block_labels.append(sentence_block_batch[i][mask_idx, :])
        sentence_block_batch[i][mask_idx, :] = sentence_block_mask_vector
        mask_indices.extend(mask_idx + num_sentence_blocks*i)
		
    sentence_block_batch = torch.stack(sentence_block_batch)

    sentence_block_labels = torch.cat(sentence_block_labels)

    return sentence_block_batch, sentence_block_labels, mask_indices