from torchmetrics import Metric

class AUPRC(Metric):
    """
    Computes the area under curve (AUC) of the receiver operator characteristic (ROC)
    Example:
        >>> pred = torch.tensor([0, 1, 2, 3])
        >>> target = torch.tensor([0, 1, 2, 2])
        >>> metric = AUROC()
        >>> metric(pred, target)
        tensor(0.3333)
    """

    def __init__(
            self,
            pos_label: int = 1,
            reduce_group: Any = None,
            reduce_op: Any = None,
    ):
        """
        Args:
            pos_label: positive label indicator
            reduce_group: the process group to reduce metric results from DDP
            reduce_op: the operation to perform for ddp reduction
        """
        super().__init__(name='auroc',
                         reduce_group=reduce_group,
                         reduce_op=reduce_op)

        self.pos_label = pos_label

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor,
            sample_weight: Optional[Sequence] = None
    ) -> torch.Tensor:
        """
        Actual metric computation
        Args:
            pred: predicted labels
            target: groundtruth labels
            sample_weight: the weights per sample
        Return:
            torch.Tensor: classification score
        """
        return auroc(pred=pred, target=target,
                     sample_weight=sample_weight,
                     pos_label=self.pos_label)
