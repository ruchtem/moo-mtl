# datsets override methods override generic

#
# datasets
#
adult = dict(
    dataset='adult',
    dim=(88,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    reference_point=[2, 2],
    epochs=50,
    use_scheduler=False,
)

credit = dict(
    dataset='credit',
    dim=(90,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    reference_point=[2, 2],
    epochs=50,
    use_scheduler=False,
)

compas = dict(
    dataset='compas',
    dim=(20,),
    objectives=['BinaryCrossEntropyLoss', 'deo'],
    reference_point=[2, 2],
    epochs=50,
    use_scheduler=False,
)

multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
    scheduler_milestones=[15,30,45,60,75,90],
)


multi_fashion = dict(
    dataset='multi_fashion',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
    scheduler_milestones=[15,30,45,60,75,90],
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
    scheduler_milestones=[15,30,45,60,75,90],
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    task_ids=list(range(40)),
    objectives=['BinaryCrossEntropyLoss' for _ in range(40)],
    reference_point=[2, 2],
    n_test_rays=100,
    scheduler_milestones=[15,30],
    train_eval_every=0,     # do it in parallel manually
    eval_every=0,
    model_name='efficientnet-b3',   #'resnet18', 
)

#
# methods
#
paretoMTL = dict(
    method='ParetoMTL',
    lr=1e-3,
    batch_size=256,
    epochs=100,
    num_starts=5,
)

cosmos = dict(
    method='cosmos',
    lr=1e-3,
    batch_size=256,
    epochs=150,
    num_starts=1,
    early_fusion=True,
    late_fusion=False,
    alpha_generator_dim=2,
    alpha_dir=0.2,   # dirichlet sampling, None=Uniform sampling
    train_eval_every=2,
    internal_solver='linear', # 'epo' or 'linear'
)

SingleTaskSolver = dict(
    method='SingleTask',
    lr=1e-3,
    batch_size=256,
    epochs=150,
    num_starts=2,
)

hyperSolver = dict(
    method='hyper',
    lr=1e-4,
    batch_size=256,
    epochs=150,     # 100 for multi_mnist
    num_starts=1,
    alpha_dir=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='epo', # 'epo' or 'linear'
)

#
# Common settings
#
generic = dict(
    logdir='results',
    num_workers=4,  # dataloader worker threads
    n_test_rays=25,
    eval_every=1,
    train_eval_every=0, # 0 for not evaluating on the train set
    use_scheduler=True,
)
