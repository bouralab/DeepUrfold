from torchmetrics.functional.classification import auc_decorator, precision_recall_curve

def auprc(
        pred: torch.Tensor,
        target: torch.Tensor,
        sample_weight: Optional[Sequence] = None,
        pos_label: int = 1.,
) -> torch.Tensor:
    """
    Compute Area Under the Preciosion Recal Curve (ROC PRC) from prediction scores
    Args:
        pred: estimated probabilities
        target: ground-truth labels
        sample_weight: sample weights
        pos_label: the label for the positive class
    Return:
        Tensor containing ROCPRC score
    Example:
        >>> x = torch.tensor([0, 1, 2, 3])
        >>> y = torch.tensor([0, 1, 2, 2])
        >>> auroc(x, y)
        tensor(0.3333)
    """

    @auc_decorator(reorder=True)
    def _auprc(pred, target, sample_weight, pos_label):
        return precision_recall_curve(pred=pred, target=target,
                                      sample_weight=sample_weight,
                                      pos_label=self.pos_label)

    return _auprc(pred=pred, target=target, sample_weight=sample_weight, pos_label=pos_label)
