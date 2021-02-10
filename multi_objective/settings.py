# datsets override methods override generic

#
# datasets
#
adult = dict(
    dataset='adult',
    dim=(88,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    lamda=.01,
    alpha=.5,
)

credit = dict(
    dataset='credit',
    dim=(90,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    lamda=.01,
    alpha=[.1, .5],
)

compas = dict(
    dataset='compas',
    dim=(20,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    epochs=50,
    use_scheduler=False,
    lamda=.01,
    alpha=.5,
)

multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=2,
    alpha=1.2,
)

multi_fashion = dict(
    dataset='multi_fashion',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=2,
    alpha=1.2,
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=8,
    alpha=1.2,
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    task_ids=[25, 27],      # task_ids=[16, 22],
    objectives=['BinaryCrossEntropyLoss' for _ in range(2)],
    n_test_rays=100,
    epochs=100,
    use_scheduler=False,
    train_eval_every=0,     # do it in parallel manually
    eval_every=0,
    model_name='efficientnet-b4',
    lr=0.0005,
    logdir='results_celeba',
    lamda=3,
    alpha=1,
    checkpoint_every=1,
    batch_size=32,
)

#
# methods
#
paretoMTL = dict(
    method='ParetoMTL',
    num_starts=5,
    scheduler_gamma=0.5,
    scheduler_milestones=[15,30,45,60,75,90],
)

cosmos = dict(
    method='cosmos',
    lamda=2,        # Default for multi-mnist
    alpha=1.2,      #
)

mgda = dict(
    method='mgda',
    lr=1e-4,
    approximate_norm_solution=True,
    normalization_type='loss+',
    use_scheduler=False,
)

SingleTaskSolver = dict(
    method='SingleTask',
    num_starts=2,   # two times for two objectives (sequentially)
)

uniform_scaling = dict(
    method='uniform',
)

hyperSolver_ln = dict(
    method='hyper_ln',
    lr=1e-4,
    epochs=150,
    alpha=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='linear', # 'epo' or 'linear'
)

hyperSolver_epo = dict(
    method='hyper_epo',
    lr=1e-4,
    epochs=150,
    alpha=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='epo',
)

#
# Common settings
#
generic = dict(    
    # Seed.
    seed=1,
    
    # Directory for logging the results
    logdir='results',

    # dataloader worker threads
    num_workers=4,

    # Number of test preference vectors for Pareto front generating methods    
    n_test_rays=25,

    # Evaluation period for val and test sets (0 for no evaluation)
    eval_every=5,

    # Evaluation period for train set (0 for no evaluation)
    train_eval_every=0,

    # Checkpoint period (0 for no checkpoints)
    checkpoint_every=10,

    # Use a multi-step learning rate scheduler with defined gamma and milestones
    use_scheduler=True,
    scheduler_gamma=0.1,
    scheduler_milestones=[20,40,80,90],

    # Number of train rays for methods that follow a training preference (ParetoMTL and MGDA)
    num_starts=1,

    # Training parameters
    lr=1e-3,
    batch_size=256,
    epochs=100,

    # Reference point for hyper-volume calculation
    reference_point=[2, 2],
)
