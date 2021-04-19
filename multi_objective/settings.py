import random
# datsets override methods override generic

#
# datasets
#
adult = dict(
    dataset='adult',
    dim=(88,),
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    epochs=50,
    # lr_scheduler='MultiStep',
    lamda=0,
    alpha=1,
    lr=1e-3,
    eval_every=1,
)

credit = dict(
    dataset='credit',
    dim=(90,),
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    epochs=50,
    use_scheduler=False,
    lamda=5,
    lr=1e-3,
    alpha=.5,
    eval_every=1,
    batch_size=1024,
)

compass = dict(
    dataset='compass',
    dim=(20,),
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    epochs=50,
    use_scheduler=False,
    lamda=.01,
    alpha=.5,
)

multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    task_ids=['l', 'r'],
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=3,
    alpha=1,
    eval_every=5,
    lr_scheduler='MultiStep',
)

multi_fashion = dict(
    dataset='multi_fashion',
    dim=(1, 36, 36),
    task_ids=['l', 'r'],
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=2,
    alpha=1.2,
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    task_ids=['l', 'r'],
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=3,
    alpha=1,
    eval_every=1,
    lr_scheduler='MultiStep',
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    # task_ids=[16, 22],                                    # easy tasks
    # task_ids=[25, 27],                                    # hard tasks
    # task_ids=[16, 22, 24],                                # 3 objectives
    task_ids=[16, 22, 24, 26],                            # 4 objectives
    # task_ids=[random.randint(0, 39) for _ in range(10)],  # 10 random tasks
    # task_ids=list(range(40)),                             # all tasks
    n_partitions=3,
    objectives=['BinaryCrossEntropyLoss' for _ in range(40)],
    reference_point=[1 for _ in range(4)],
    epochs=100,
    lr_scheduler='MultiStep',
    train_eval_every=0,     # do it in parallel manually
    eval_every=5,           #
    model_name='efficientnet-b4',   # we also experimented with 'resnet-18', try it.
    lr=0.0001,
    lamda=1,
    alpha=.5,
    checkpoint_every=1,
    batch_size=32,
)

cityscapes = dict(
    dataset='cityscapes',
    dim=(3, 256, 512), # height width
    task_ids=['segm', 'inst', 'depth'],
    objectives=['CrossEntropyLoss', 'L1Loss', 'L1Loss'],
    metrics=['mIoU', 'L1Loss', 'L1Loss'],
    batch_size=8,
    epochs=200,
    n_partitions=5,
    lamda=.5,
    lr=0.0005,
    eval_every=3,
    approximate_norm_solution=True,
    normalization_type='l2',
    lr_scheduler='MultiStep',
    alpha=1.3,
)

coco = dict(
    dataset='coco',
    dim=(3, 512, 512),   # max dims as images have different sizes
    task_ids=['classifier', 'box_reg', 'mask'],
    objectives=['void' for _ in range(6)],
    metrics=['mIoU', 'L1Loss', 'L1Loss', 'mIoU'],
    batch_size=2,
    device='cpu',
    num_workers=0,
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
    approximate_norm_solution=False,
    normalization_type='none',
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

    # Which results to generate during evaluation.
    # If 'pareto_front' is selected, it will be obtained by using n_test_rays
    # eval_mode=['center_ray', 'pareto_front'],

    # Number of test preference vectors for Pareto front generating methods along one axis (minus one corner point)
    n_partitions=24,

    # Evaluation period for val and test sets (0 for no evaluation)
    eval_every=5,

    # Evaluation period for train set (0 for no evaluation)
    train_eval_every=0,

    # Checkpoint period (0 for no checkpoints)
    checkpoint_every=10,

    # One of [None, CosineAnnealing, MultiStep]
    lr_scheduler = None,
    scheduler_milestones=None,

    # Number of train rays for methods that follow a training preference (ParetoMTL and MGDA)
    num_starts=1,

    # Training parameters
    lr=1e-3,
    batch_size=256,
    epochs=100,

    # Reference point for hyper-volume calculation
    reference_point=[2, 2],

    # cuda or cpu
    device='cuda',

    # use defaults
    metrics=None,
)
