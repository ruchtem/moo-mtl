
adult = dict(
    dataset='adult',
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    reference_point=[1, 1],
)

credit = dict(
    dataset='credit',
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    reference_point=[1, 1],
)

compas = dict(
    dataset='compas',
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
    reference_point=[1, 1],
)

multi_mnist = dict(
    dataset='multi_mnist',
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
)


multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
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
    late_fusion=True,
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
    epochs=150,
    num_starts=1,
    warmstart=False,
    alpha_dir=.2,   # dirichlet sampling
)

generic = dict(
    logdir='results'
)
