import torch
import numpy as np


def mcr(logits, labels, sensible_attribute=None):
    # missclassification rate
    with torch.no_grad():
        if logits.shape[1] == 1:
            # binary case
            logits = torch.squeeze(logits)
            y_hat = torch.round(torch.sigmoid(logits))
        else:
            y_hat = torch.argmax(logits, dim=1)
        accuracy = sum(y_hat == labels) / len(y_hat)
    return 1 - accuracy.item()


def DDP(logits, labels, sensible_attribute):
    """Difference in Democratic Parity"""
    with torch.no_grad():
        n = logits.shape[0]
        logits_s_negative = logits[sensible_attribute.bool()]
        logits_s_positive = logits[~sensible_attribute.bool()]

        return torch.abs(1/n*sum(logits_s_negative > 0) - 1/n*sum(logits_s_positive > 0).item()).cpu().item()