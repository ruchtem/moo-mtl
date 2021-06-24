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

_C.accuracy_class = None
_C.sensible_attribute = None


#
# Model
#

_C.channel_multiplier = 1.


#
# Training
#

# Number of epochs to train
_C.epochs = 10

# dataloader worker threads
_C.num_workers = 4

# Checkpoint period per epoch (0 for no checkpoints)
_C.checkpointing = False

# We support 2 optional learing rate schedulers: 'MultiStep' and 'Cosine'. MultiStep is decaying 
# the learing rate with 0.1 at 0.33 * epochs and 0.66 * epochs.
_C.lr_scheduler = 'none'

# learning rate
_C.lr = 1e-3

# weight decay (l2 regularization)
_C.weight_decay = 1e-3

# batch size
_C.batch_size = 256

# method to generate pareto front. One of 'cosmos', 'phn', 'mgda', 'single_task', 'uniform'
_C.method = 'cosmos'

#
# Method specific
#

# Pareto Multi-Task Learning
#
# Lin, X., Zhen, H. L., Li, Z., Zhang, Q., & Kwong, S. (2019).
# Pareto multi-task learning. arXiv preprint arXiv:1912.12854.

# Number of points on the pareto front (i.e. num of models)
_C.num_models = 5


# Multiple Gradient Descent Algorithm

# Sener, O., & Koltun, V. (2018). Multi-task learning as 
# multi-objective optimization. arXiv preprint arXiv:1810.04650.

# Use the approximation
_C.approximate_mgda = False

# Gradient normalization. One of 'none', 'loss', 'loss+', 'l2'
# Added 'init_loss' proposed in Milojkovic et. al.
_C.normalization_type='none'


# Pareto HyperNetworks
#
# Navon, A., Shamsian, A., Chechik, G., & Fetaya, E. (2020). 
# Learning the Pareto Front with Hypernetworks. arXiv preprint arXiv:2010.04104.

# Dirichlet sampling
_C.alpha=.2

# 'epo' or 'linear'
_C.internal_solver_phn='linear'


# COSMOS
#

_C.n_train_partitions_cosmos = 8
_C.loss_mins = [0.]  # will be repeated for all losses
_C.loss_maxs = [1.]

_C.lambda_clipping = 5.
_C.lambda_lr = 0.2
_C.dampening = 0.2


_C.upsample_ratio = 1.


# cosmos and pmtl
_C.train_ray_mildening = 0.0


# NSGA-II
_C.population_size = 100
_C.n_offsprings = 20


# single task
# If None, than train and evaluate all models simulataniously. This does not allow for 
# different hyperparemters, e.g. learning rate or weight decay for different tasks. To
# optimize those set a task id here and optimize just one model.
_C.task_id = None


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

# Evaluation period for train and test set (0 for no evaluation)
_C.train_eval_every=0
_C.test_eval_every=0

# Reference point for hyper-volume calculation
_C.reference_point=[1, 1]

# cuda or cpu
_C.device='cuda'

# use defaults
_C.metrics=None



# VAE model parameters
# taken from https://github.com/swisscom/ai-research-mamo-framework/blob/master/models/params_multi_VAE.yaml
_C.vae_params = CN()
_C.vae_params.dropout = 0.5
_C.vae_params.no_latent_features = 200
_C.vae_params.norm_mean = 0.0
_C.vae_params.norm_std = 0.001
_C.vae_params.input_size = 1000
_C.vae_params.output_size = 1000
_C.vae_params.enc1_out = 600
_C.vae_params.enc2_in = 600
_C.vae_params.enc2_out = 400
_C.vae_params.dec1_in = 200
_C.vae_params.dec1_out = 600
_C.vae_params.dec2_in = 600

# Required to weight the vae loss with prices for movielens
_C.loss_weights = 'None'

# Annealing of KL regularization
_C.beta_start = 0
_C.beta_cap = 0.3
_C.beta_step = 0.3/10000

# For recommender system metrics
_C.K = 20


def get_cfg_defaults():
    """Get a CfgNode object with default values for my_project."""
    return _C.clone()