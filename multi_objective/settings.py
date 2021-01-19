
adult = dict(
    dataset='adult',
    objectives=['BinaryCrossEntropyLoss', 'ddp'],
)

multi_mnist = dict(
    dataset='multi_mnist',
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss']
)


multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss']
)

paretoMTL = dict(
    method='ParetoMTL',
    lr=1e-3,
    batch_size=64,
    epochs=5,
    num_starts=5,
    warmstart=False
)

afeature = dict(
    method='afeature',
    lr=1e-4,
    batch_size=64,
    epochs=300,
    num_starts=1,
    warmstart=True,
    early_fusion=True
)

SingleTaskSolver = dict(
    method='SingleTask',
    lr=1e-4,
    batch_size=64,
    epochs=100,
    num_starts=1,
    warmstart=False,
    task=1,
)

hyperSolver = dict(
    method='hyper',
    lr=1e-4,
    batch_size=256,
    epochs=150,
    num_starts=1,
    warmstart=False,
)


