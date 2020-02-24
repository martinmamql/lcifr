# Learning Certified Individually Fair Representations <a href="https://www.sri.inf.ethz.ch/"><img width="100" alt="portfolio_view" align="right" src="http://safeai.ethz.ch/img/sri-logo.svg"></a>

LCIFR is a state-of-the art system for training neural networks with
 provable certificates of individual fairness.
LCIFR leverages the theoretical framework introduced by
 [McNamara et al.](https://arxiv.org/abs/1710.04394) which partitions the
 task of learning fair representations into three parties:
- **data regulator**: defines a fairness property for the particular task at
 hand
- **data producer**: processes sensitive user data and transforms it into a
 latent representation
- **data consumer**: performs predictions based on the new representation

The key idea behind LCIFR is to learn a representation that provably maps
 similar individuals to latent representations at most epsilon apart in
 l-infinity distance, enabling data consumers to certify individual fairness by
 proving epsilon-robustness of their classifier.
Furthermore, LCIFR allows data regulators to define rich similarity notions via logical
 constraints.
 
This implementation of LCIFR can be used as a library compatible with
 PyTorch and contains all code, datasets and preprocessing pipelines
  necessary to reproduce the results from our paper.
This system is developed at the
 [SRI Lab, Department of Computer Science, ETH Zurich](https://www.sri.inf.ethz.ch)
 as part of the [Safe AI project](http://safeai.ethz.ch).

## Setup Instructions

Clone this repository and its dependencies
```bash
$ git clone --recurse-submodules https://github.com/eth-sri/lcifr.git
```

Create a [conda](https://www.anaconda.com/distribution/#download-section)
 environment with the required packages
```bash
$ conda env create -f environment.yml
```

We use the GUROBI solver for certification. To run our code, apply for and
 download an [academic GUROBI License](https://www.gurobi.com/academia/academic-program-and-licenses).

## Structure

```
.
├── README.md                       - this file
├── code
│   ├── attacks                     - for adversarial training
│   ├── constraints                 - data regulator: individual fairness notions
│   ├── datasets                    - downloads and preprocesses datasets
│   ├── experiments
│   │   ├── args_factory.py         - defines training parameters
│   │   ├── certify.py              - runs end-to-end certification
│   │   ├── train_classifier.py     - data consumer: trains model
│   │   └── train_encoder.py        - data producer: learns representation
│   ├── metrics                     - group/individual fairness metrics
│   ├── models                      - model architectures
|   └── utils
├── data                            - created when running experiments
├── dl2                             - dependency
├── models                          - created when running experiments
├── logs                            - created when running experiments
├── results                         - created when running experiments
|── environment.yml                 - conda environment
└── setup.sh                        - activates conda environment and sets paths
```

Some files omitted.

## Reproducing the Experiments

Activate the conda environment and set the PYTHONPATH
```bash
$ source setup.sh
```

Enter the experiment directory
```bash
$ cd code/experiments
```

Run the end-to-end framework for all constraints
```bash
$ ./noise.sh
$ ./cat.sh
$ ./cat_noise.sh
$ ./attribute.sh
$ ./quantiles.sh
```

Run the end-to-end framework on a large network
```bash
$ ./scaling.sh
```

Run the end-to-end framework for transfer learning
```bash
$ ./transfer.sh
```

Once started, the training progress can be monitored in Tensorboard with
```bash
$ tensorboard --logdir logs
```

## Citing This Work

```
@incollection{ruoss2020learning,
    title = {Learning Certified Individually Fair Representations},
    author = {Ruoss, Anian and Balunovic, Mislav and Fischer, Marc and Vechev, Martin},
    year = {2020}
}	
```

## Contributors

* Anian Ruoss (anruoss@ethz.ch)
* [Mislav Balunović](https://www.sri.inf.ethz.ch/people/mislav)
* [Marc Fischer](https://www.sri.inf.ethz.ch/people/marc)
* [Martin Vechev](https://www.sri.inf.ethz.ch/people/martin)

## License and Copyright

* Copyright (c) 2020 [Secure, Reliable, and Intelligent Systems Lab (SRI), ETH Zurich](https://www.sri.inf.ethz.ch)
* Licensed under the [Apache License](http://www.apache.org/licenses)
