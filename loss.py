import torch
import torch.nn as nn  

class TriBCE_Loss(nn.Module):
    def __init__(self):
        super(TriBCE_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true, y_d, y_s):
        loss = self.bce_loss(y_pred, y_true)
        loss_d = self.bce_loss(y_d, y_true)
        loss_s = self.bce_loss(y_s, y_true)
        weight_d = loss_d - loss
        weight_s = loss_s - loss
        weight_d = torch.where(weight_d > 0, weight_d, torch.zeros(1).to(weight_d.device))
        weight_s = torch.where(weight_s > 0, weight_s, torch.zeros(1).to(weight_s.device))
        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss

class Weighted_TriBCE_Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(Weighted_TriBCE_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true, y_d, y_s):
        pos_count = torch.sum(y_true)
        neg_count = y_true.numel() - pos_count

        pos_count = torch.clamp(pos_count, min=1.0)
        neg_count = torch.clamp(neg_count, min=1.0)

        pos_weight = neg_count / (pos_count + neg_count)
        neg_weight = pos_count / (pos_count + neg_count)

        sample_weight = torch.where(y_true == 1, pos_weight, neg_weight)

        bce_loss = nn.BCELoss(weight=sample_weight, reduction=self.reduction)

        loss = bce_loss(y_pred, y_true)
        loss_d = bce_loss(y_d, y_true)
        loss_s = bce_loss(y_s, y_true)

        weight_d = torch.where(loss_d > loss, loss_d - loss, torch.zeros_like(loss_d))
        weight_s = torch.where(loss_s > loss, loss_s - loss, torch.zeros_like(loss_s))

        loss = loss + loss_d * weight_d + loss_s * weight_s
        return loss
    
class BCE_Loss(nn.Module):
    def __init__(self):
        super(BCE_Loss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, y_pred, y_true):
        return self.bce_loss(y_pred, y_true)

class Weighted_BCE_Loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(Weighted_BCE_Loss, self).__init__()
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        pos_count = torch.sum(y_true)
        neg_count = y_true.numel() - pos_count

        pos_count = torch.clamp(pos_count, min=1.0)
        neg_count = torch.clamp(neg_count, min=1.0)

        pos_weight = neg_count / (pos_count + neg_count)
        neg_weight = pos_count / (pos_count + neg_count)

        sample_weight = torch.where(y_true == 1, pos_weight, neg_weight)

        bce_loss = nn.BCELoss(weight=sample_weight, reduction=self.reduction)
        return bce_loss(y_pred, y_true)