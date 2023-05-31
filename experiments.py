import ray
from ray import air, tune
from ray.tune.schedulers import *
from ray.tune.search import * 
from ray.tune.experiment.trial import Trial
from ray.tune.logger import pretty_print

SCHEDULERS = {
    "fifo": True,
    "median_stopping_rule": True,
    "hyperband": True,
    "async_hyperband": True,
    "hb_bohb": False,
    "pbt": False,
    "pbt_replay": False,
    "pb2": False,
    "resource_changing": False
}

OPTIMIZERS = {
    #"random_search": True, 
    "variant_generator": True, 
    "random": True, 
    "ax": True,
    "dragonfly": True,
    "skopt": True,
    "hyperopt": True,
    "bayesopt": True,
    "bohb": True,
    "nevergrad": True,
    "optuna": True,
    "zoopt": True,
    "sigopt": True,
    "hebo": True,
    "blendsearch": True,
    "cfo": True,
}

def log_ten(start=1, steps=5):
    interval = []
    for i in range(steps):
        interval.append(start)
        start /= 10
    return interval

def run_sweep(schedulers=SCHEDULERS, optimizers=OPTIMIZERS, sweeps={'schedulers': False, 'optimizers': True}, sweep_name="test"):
    """Benchmark scheduler and optimizer perforamce"""

    if sweeps['schedulers']:
        schedulers_sweep = []
        for scheduler in schedulers:
            if schedulers[scheduler]:
                schedulers_sweep.append(scheduler)
    else:
        schedulers_sweep = ["fifo"]

    if sweeps['optimizers']:
        optimizers_sweep = []
        for optimizer in optimizers:
            if optimizers[optimizer]:
                optimizers_sweep.append(optimizer)
    else:
        optimizers_sweep = ["random_search"]

    if True: 
        def name(trial: Trial):
            return str(trial.config["scheduler"])

        ray.shutdown()
        ray.init()
        trainable = "PPO"
        stopping_criteria = {"training_iteration": 10}
        tuner = tune.Tuner(
            trainable=trainable,
            run_config=air.RunConfig(
                local_dir="./results",
                name=sweep_name,
                stop=stopping_criteria,
            ),
            tune_config=tune.TuneConfig(
                mode="min",
                num_samples=1,
                trial_name_creator=name,
            ),
            param_space={
                "env": "CartPole-v1",
                "num_workers": 1, #tune.grid_search([1, 2, 4]),
                "num_cpus": 1,
                "num_gpus": 0,
                "lr": 1, #tune.grid_search(log_ten(1, 5))
                'hyperparam_mutations': 0,
                "scheduler": tune.grid_search(schedulers_sweep),
                "search_alg": tune.grid_search(optimizers_sweep),
            }
        )
        results = tuner.fit()
        ray.shutdown()
    return