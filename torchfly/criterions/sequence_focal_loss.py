import torch
import torch.nn as nn
import torch.nn.functional as F

# pylint:disable=no-member

class SequenceFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, logits, targets, weights, label_smoothing=-1, reduce=True):
        
        # shape : (batch * sequence_length, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # shape : (batch * sequence_length, num_classes)
        probs_flat = F.softmax(logits_flat, dim=-1)
        # shape : (batch * max_len, 1)
        targets_flat = targets.view(-1, 1).long()   
        
        if label_smoothing > 0.0:
            num_classes = logits.size(-1)
            smoothing_value = label_smoothing / num_classes
            # Fill all the correct indices with 1 - smoothing value.
            one_hot_targets = torch.zeros_like(probs_flat).scatter_(-1, targets_flat, 
                                                                        1.0 - label_smoothing)
            smoothed_targets = one_hot_targets + smoothing_value
            
            # focal loss
            fl_flat = - self.alpha * (1 - probs_flat).pow(self.gamma) * \
                            torch.log(probs_flat) * smoothed_targets
            
            fl_flat = fl_flat.sum(-1, keepdim=True)
            #negative_log_likelihood_flat = negative_log_likelihood_flat.sum(-1, keepdim=True)
        else:
            # select only target index probability
            cross_probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
            fl_flat = - self.alpha * (1 - cross_probs_flat).pow(self.gamma) * \
                            torch.log(cross_probs_flat)
        
        # shape : (batch, sequence_length)
        fl = fl_flat.view(targets.size(0), targets.size(1))
        # shape : (batch, sequence_length)
        fl = fl * weights.float()
        
        # shape : (batch_size,)
        loss = fl.sum(1) / (weights.sum(1) + 1e-13)

        if bool(reduce):
            # scalar
            loss = loss.mean()
            
        return loss