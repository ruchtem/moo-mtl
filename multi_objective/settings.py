
adult = dict(
    dataset='adult',
    dim=(88,),
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    reference_point=[1, 1],
)

credit = dict(
    dataset='credit',
    dim=(90,),
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    reference_point=[1, 1],
)

compas = dict(
    dataset='compas',
    dim=(20,),
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    reference_point=[1, 1],
)

multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
)


multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    task_ids=[22, 39],   # list(range(40)) for all tasks
    objectives=['BinaryCrossEntropyLoss', 'BinaryCrossEntropyLoss'],
    reference_point=[2, 2],
)

paretoMTL = dict(
    method='ParetoMTL',
    lr=1e-3,
    batch_size=256,
    epochs=5,
    num_starts=5,
    warmstart=False
)

afeature = dict(
    method='afeature',
    lr=1e-4,
    batch_size=256,
    epochs=150,
    num_starts=1,
    warmstart=True,
    early_fusion=True,
    late_fusion=False,
    alpha_generator_dim=2,
    alpha_dir=.2,   # dirichlet sampling
)

SingleTaskSolver = dict(
    method='SingleTask',
    lr=1e-4,
    batch_size=256,
    epochs=100,
    num_starts=1,
    warmstart=False,
    task=0,
)

hyperSolver = dict(
    method='hyper',
    lr=1e-4,
    batch_size=256,
    epochs=100,     # 100 for multi_mnist
    num_starts=1,
    warmstart=False,
    alpha_dir=.2,   # dirichlet sampling
)

generic = dict(
    logdir='results',
    num_workers=4,  # dataloader worker threads
    n_test_rays=25,
    eval_every=10,
)
