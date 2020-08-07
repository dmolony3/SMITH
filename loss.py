import torch

def masked_sentence_block_loss(predicted_embeddings, label_embeddings):
    """Computes the masked sentence block loss"""

    sim = torch.tensordot(predicted_embeddings.unsqueeze(1), label_embeddings.T.unsqueeze(0), dims=2)
    B = sim.shape[0]
	
    mask = torch.eye(B).long()

    p = torch.exp(sim)/torch.sum(torch.exp(sim), 1)

    log_likelihood = torch.log(p)*mask

    negative_log_likelihood = -torch.sum(log_likelihood)/B

    return negative_log_likelihood