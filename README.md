# Learned Query Optimizer


## About The Project

*Query Optimization is considered to be one of the most difficult challenges in query
processing as it falls into the class of NP-hard problems. The rise of deep learning and deep reinforcement
learning, in particular, has contributed to many scientific and industrial fields, including database management and
join query optimization. 
In this project, we study Join Order Optimization with reinforcement learning. For this purpose, we have developed 
a learned query optimizer based on a combination of the [Neo approach](https://www.vldb.org/pvldb/vol12/p1705-marcus.pdf), 
Tree-Transformers and reinforcement learning heuristics.
The method was tested with [Join Order Benchmark](https://github.com/gregrahn/join-order-benchmark) on two database systems: PostgreSQL and Huawei GaussDB.*



## Installation

#### Learned Query Optimizer:

The recommended way is to use a docker. `Dockerfile` is in the root directory.

#### Database:

We pass the desired order of joins to the execution engine with the help of the [plan hinting](https://pghintplan.osdn.jp/pg_hint_plan.html) mechanism.
In GaussDB and enterprise versions of PostgreSQL this plugin is already preinstalled, 
but for the basic PostgreSQL version you will need to install it manually. 
Other database systems may lack such a way of controlling execution plans.


## Usage

First, make sure that the database is running and accepts connections.

#### Training

Optimizer training can be run as follows:

```
python run_experiment.py <config_path> <database_env_config_path>
```

- `config_path`: the path to the settings of the model and the learning algorithm in yaml format

```yaml
neo_args:
  logdir: str # the directory where to save the logs
  baseline_path: str # the directory with plan examples (e.g. from the optimizer built into the db engine) 
  latency: bool # if true, learn from real execution times, otherwise from costs estimated by the db engine.
  n_workers: int # number of workers.
  num_complete_plans: int # number of generated plans for the query at each algorithm step.
  sync: bool # if false, the process of searching and evaluating plans will not wait for the completion of the network update.
  total_episodes: int # number of algorithm steps.
  val_size: int # the size of the validation subset of partial plans.
  encoding: str # the type of plan encoding.
  cost_func: str # the type of cost function.
  reward_weighting: bool # if true, scale rewards depending on the query size.
  cardinality: str # add log cardinalities to plan encoding.
  eps: float # probabilty of choosing random join instead of predicted.
  device: str # cpu or cuda device.
net_args:
  pretrained_path: str # the path to the saved value network model.
  fit_pretrained_layers: list # list of modules that should be trained.
  # other parameters specific for model, such as number of layers, dimensions, etc.
train_args:
  epochs : int # number of epochs
  min_iters: int # minimal num of training steps
  batch_size: int # batch size 
  betas: tuple # betas argument in torch.optim.Adam
  lr: float # learning rate
  scheduler: float # learning rate decay in the scheduler
```
- `database_env_config_path`: the path to the database parameters in json format

```json
{
  "psycopg_connect_url": "db connection string",
  "db": "database type ('gaussdb' or 'postgres')",
  "join_types": "list of allowed join methods",
  "test_queries": "list of queries that will be excluded during the training",
  "scheme": "db scheme",
  "db_data": "list with the following information for each query: tables, aliases, join conditions, original query string (query without specified join order)"
}
```

See the `config` folder for examples of configs. 

#### Evaluating

See `generate_and_evaluate.ipynb` for an example of generating plans and measuring their execution time.


## Tests
All test can be used with pytest.
For example:
```
POSTGRES_CONNECT_URL=postgres://imdb:pwdpwd@127.0.0.1:5432/imdb pytest ./tests/
```