import torch

EPSILON = 1e-20


def classwise_iou(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    """
    # dims = (0, *range(2, len(output.shape)))
    dims = (-1, -2) if output.dim() == 2 else (-1, -2, -3)
    # gt = torch.zeros_like(output).scatter_(1, gt[:, None, :], 1)
    intersection = output * gt
    union = output + gt - intersection
    classwise_iou = (intersection.sum(dim=dims).float() +
                     EPSILON) / (union.sum(dim=dims) + EPSILON)

    return classwise_iou


def classwise_f1(output, gt):
    """
    Args:
        output: torch.Tensor of shape (n_batch, n_classes, image.shape)
        gt: torch.LongTensor of shape (n_batch, image.shape)
    ret:
        classwise_f1: torch.Tensor of shape (n_classes)
        # AUROC: torch.Tensor of shape (n_classes)
    """

    true_positives = (((output == 1) *
                       (gt == 1)).sum()).clone().detach().float()
    selected = ((output == 1).sum()).float()
    relevant = ((gt == 1).sum()).float()

    precision = (true_positives + EPSILON) / (selected + EPSILON)
    recall = (true_positives + EPSILON) / (relevant + EPSILON)
    classwise_f1 = 2 * (precision * recall) / (precision + recall)

    # specificity = (((output == 0) * (gt == 0)).sum()).float() / ((
    #     (gt == 0).sum()).float() + EPSILON)
    # sensitivity = recall

    return classwise_f1


def make_weighted_metric(classwise_metric):
    """
    Args:
        classwise_metric: classwise metric like classwise_IOU or classwise_F1
    """

    def weighted_metric(output, gt, weights=None):

        # dimensions to sum over
        dims = (-1, -2) if output.dim() == 2 else (-1, -2, -3)

        # default weights
        if weights == None:
            weights = torch.ones(output.shape[1]) / output.shape[1]
        else:
            # creating tensor if needed
            if len(weights) != output.shape[1]:
                raise ValueError(
                    "The number of weights must match with the number of classes"
                )
            if not isinstance(weights, torch.Tensor):
                weights = torch.tensor(weights)
            # normalizing weights
            weights /= torch.sum(weights)

        classwise_scores = classwise_metric(output, gt).cpu()

        return (classwise_scores * weights).sum(dim=dims).item()

    return weighted_metric


jaccard_index = make_weighted_metric(classwise_iou)
f1_score = make_weighted_metric(classwise_f1)

if __name__ == '__main__':
    output, gt = torch.zeros(3, 2, 5, 5), torch.zeros(3, 5, 5).long()
    print(classwise_iou(output, gt))
