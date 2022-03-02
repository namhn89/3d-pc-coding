import numpy as np
import torch


def get_bce_loss(pred: torch.Tensor, label: torch.Tensor):
    """ (Weighted) Binary cross entropy loss.

    Args:
        pred (torch.Tensor): _description_
        label (torch.Tensor): _description_
    """
    occupancy = torch.clamp(torch.sigmoid(pred), min=1e-7, max=1.0 - 1e-7)
    mask_neg = torch.lt(label, 0.5)
    mask_pos = torch.gt(label, 0.5)
    occupancy_neg = torch.masked_select(occupancy, mask_neg)
    occupancy_pos = torch.masked_select(occupancy, mask_pos)
    empty_loss = torch.mean(-torch.log(1.0 - occupancy_neg))
    full_loss = torch.mean(-torch.log(occupancy_pos))

    return empty_loss, full_loss


def get_confusion_matrix(pred: torch.Tensor, label: torch.Tensor, th=0.):
    """Confusion matrix:
      1   0
    1 TP  FN
    0 FP  TN(option)

    Args:
        pred (_type_): _description_
        label (_type_): _description_
        th (_type_, optional): _description_. Defaults to 0..
    """
    pred = torch.squeeze(pred, 1)
    label = torch.squeeze(label, 1)

    pred = torch.gt(pred, th).float()
    label = torch.gt(label, th).float()

    TP = pred * label
    FP = pred * (1. - label)
    FN = (1. - pred) * label
    # TN = (1 - pred) * (1 - label)

    return TP, FP, FN


def get_classify_metrics(pred, label, th=0.):
    """ Metrics for classification
    Args:
        pred (_type_): _description_
        label (_type_): _description_
        th (_type_, optional): _description_. Defaults to 0..

    Returns:
        _type_: _description_
    """

    TP, FP, FN = get_confusion_matrix(pred, label, th=th)
    TP = torch.sum(TP)
    FP = torch.sum(FP)
    FN = torch.sum(FN)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    IoU = TP / (TP + FP + FN)

    return precision, recall, IoU


def get_bits(likelihood):
    bits = -torch.sum(torch.log2(likelihood))
    return bits


def test():
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    np.random.seed(108)
    data = np.random.rand(2, 64, 64, 64, 1) * 10 - 5
    data = data.astype("float32")
    label = np.random.rand(2, 64, 64, 64, 1)
    label[label >= 0.97] = 1
    label[label < 0.97] = 0
    label = label.astype("float32")

    data = torch.from_numpy(data).to(device)
    label = torch.from_numpy(label).to(device)

    loss1, loss2 = get_bce_loss(data, label)
    print("loss1: ", loss1)
    print("loss2: ", loss2)


if __name__ == "__main__":
    test()
