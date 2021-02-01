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
)

multi_fashion = dict(
    dataset='multi_fashion',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    reference_point=[2, 2],
    alpha_dir=.5,  # turned off for ablation
)

celeba = dict(
    dataset='celeba',
    dim=(3, 64, 64),
    task_ids=list(range(40)),
    objectives=['BinaryCrossEntropyLoss' for _ in range(40)],
    reference_point=[2, 2],
    n_test_rays=100,
    epochs=100,
    use_scheduler=False,
    train_eval_every=0,     # do it in parallel manually
    eval_every=0,
    model_name='efficientnet-b4',   #'resnet18', 'efficientnet-b4'
    alpha_generator_dim=4,
    lr=0.0005,   # taken from mgda paper
    logdir='results_celeba',
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
    scheduler_gamma=0.5,
    scheduler_milestones=[15,30,45,60,75,90],
)

cosmos_ln = dict(
    method='cosmos_ln',
    lr=1e-3,
    batch_size=256,
    epochs=100,
    num_starts=1,
    early_fusion=True,
    late_fusion=False,
    alpha_generator_dim=2,
    alpha_dir=1,   # dirichlet sampling, None=Uniform sampling
    internal_solver='linear',
    scheduler_gamma=0.1,
    scheduler_milestones=[20,40,80,90],
    penalty_weight = .5
)

mgda = dict(
    method='mgda',
    lr=1e-4,
    batch_size=256,
    epochs=100,
    num_starts=1,
    approximate_norm_solution=True,
    normalization_type='loss+',
    use_scheduler=False,
)

SingleTaskSolver = dict(
    method='SingleTask',
    lr=1e-3,
    batch_size=256,
    epochs=100,
    num_starts=2,
    scheduler_gamma=0.1,
    scheduler_milestones=[20,40,80,90],
)

uniform_scaling = dict(
    method='uniform',
    lr=1e-3,
    batch_size=256,
    epochs=100,
    num_starts=1,
    scheduler_gamma=0.1,
    scheduler_milestones=[20,40,80,90],
)

hyperSolver_ln = dict(
    method='hyper_ln',
    lr=1e-4,
    batch_size=256,
    epochs=150,
    num_starts=1,
    alpha_dir=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='linear', # 'epo' or 'linear'
)

hyperSolver_epo = dict(
    method='hyper_epo',
    lr=1e-4,
    batch_size=256,
    epochs=150,
    num_starts=1,
    alpha_dir=.2,   # dirichlet sampling
    use_scheduler=False,
    internal_solver='epo',
)

#
# Common settings
#
generic = dict(
    logdir='results_tmp',
    num_workers=4,  # dataloader worker threads
    n_test_rays=25,
    eval_every=5,
    train_eval_every=0, # 0 for not evaluating on the train set
    use_scheduler=True,
    scheduler_gamma=0.1,
    seed=1,
    checkpoint_every=10,
)