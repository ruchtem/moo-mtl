from fvcore.common.config import CfgNode as CN

_C = CN()

#
# dataset
#

# We support several datasets which can be loaded with their correspoding
# settings using the yaml file. Default is adult datset.
_C.dataset = 'adult'

# Dimension of the dataset. (Num of columns,) for tabular, (channels, height, width) for images
_C.dim = (88,)

# Objectives to optimize. For options see `objectives.py`
_C.objectives = ['BinaryCrossEntropyLoss', 'ddp']

# Required for MTL problems. The ids are used for the wiring of the objectives and metrics to
# the right task. Can be string. Here we assign just 0 and 1.
_C.task_ids = []

# Some datasets ignore some classes. This can be set here
_C.ignore_index = -100


#
# Training
#

# Number of epochs to train
_C.epochs = 10

# dataloader worker threads
_C.num_workers = 4

# Checkpoint period per epoch (0 for no checkpoints)
_C.checkpoint_every = 0

# We support 2 optional learing rate schedulers: 'MultiStep' and 'Cosine'. MultiStep is decaying 
# the learing rate with 0.1 at 0.33 * epochs and 0.66 * epochs.
_C.lr_scheduler = 'none'

# learning rate
_C.lr = 1e-3

# batch size
_C.batch_size = 256

#
# Methods
#

# Pareto Multi-Task Learning
#
# Lin, X., Zhen, H. L., Li, Z., Zhang, Q., & Kwong, S. (2019).
# Pareto multi-task learning. arXiv preprint arXiv:1912.12854.
_C.pmtl = CN()

# Number of points on the pareto front (i.e. num of models)
_C.pmtl.num_starts = 5

_C.pmtl.lr_scheduler = _C.lr_scheduler
_C.pmtl.lr = _C.lr


# Multiple Gradient Descent Algorithm

# Sener, O., & Koltun, V. (2018). Multi-task learning as 
# multi-objective optimization. arXiv preprint arXiv:1810.04650.
_C.mgda = CN()


# Use the approximation by Sener and Koltun
_C.mgda.approximate_norm_solution = False

# Gradient normalization. One of 'none', 'loss', 'loss+', 'l2'
_C.mgda.normalization_type='none'

_C.mgda.lr_scheduler = _C.lr_scheduler
_C.mgda.lr = _C.lr


# Pareto HyperNetworks
#
# Navon, A., Shamsian, A., Chechik, G., & Fetaya, E. (2020). 
# Learning the Pareto Front with Hypernetworks. arXiv preprint arXiv:2010.04104.
_C.phn = CN()

# Dirichlet sampling
_C.phn.alpha=.2

# 'epo' or 'linear'
_C.phn.internal_solver='linear'

_C.phn.lr_scheduler = _C.lr_scheduler
_C.phn.lr = _C.lr


# Single Task
#
_C.single_task = CN()

# This is dataset specific.
_C.single_task.task_id = None

_C.single_task.lr_scheduler = _C.lr_scheduler
_C.single_task.lr = _C.lr

# COSMOS
#
_C.cosmos = CN()

# Diriclet sampling parameter
_C.cosmos.alpha = [1.]

# Cosine similarity parameter
_C.cosmos.lamda = 1.

# Whether to normalize the losses to be on the same scale as alphas
# Handle with care, highly affects setting of the other hyperparameters.
_C.cosmos.normalize = True
_C.cosmos.instances = False

_C.cosmos.lr_scheduler = _C.lr_scheduler
_C.cosmos.lr = _C.lr

#
# Misc
#

_C.seed = 1

# Directory for logging the results
_C.logdir = 'results'

# Number of test preference vectors for Pareto front generating methods along one axis (minus one corner point)
_C.n_partitions=24

# Evaluation period for val and test sets (0 for no evaluation)
_C.eval_every=5

# Evaluation period for train set (0 for no evaluation)
_C.train_eval_every=0

# Reference point for hyper-volume calculation
_C.reference_point=[2, 2]

# cuda or cpu
_C.device='cuda'

# use defaults
_C.metrics=None


def get_cfg_defaults():
    """Get a CfgNode object with default values for my_project."""
    return _C.clone()