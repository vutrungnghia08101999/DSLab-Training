import torch


def cross_entropy(preds: torch.tensor, labels: torch.tensor) -> torch.float:
    """compute cross entropy loss between pred and gt

    Arguments:
        preds {torch.tensor} -- batch_size x 20
        labels {torch.tensor} -- batch_size x 20

    Returns:
        torch.float
    """
    s = -1.0 * labels * torch.log(preds)
    s = s.sum(dim=1)
    return s.mean()