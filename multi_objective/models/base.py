import torch


class BaseModel(torch.nn.Module):
    """
    A base class a model can inherit from.

    This is not strictly enforced as it would not allow to inherit from more complex models.
    However, double check in such cases you define all attributes correctly (i.e. task-specific
    logits).
    """


    def __init__(self, task_ids=None) -> None:
        super().__init__()
        if task_ids is None:
            print("WARNING: Cannot define task_specific logits. Assume we have just one output for all objectives!")
        else:
            self.task_ids = task_ids


    def change_input_dim(self, dim):
        raise NotImplementedError("Please add a method to your model that allows the optimizer to change its input channels.")