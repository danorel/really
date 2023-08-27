# really

really is a Python project for dealing with reinforcement learning method implementation

## Installation & Contributing

Use the [DEVELOPER.md](./DEVELOPER.md) guide to run or contribute to the project.

## Usage

a. Q-Learning algorithm

1. Search hyperparameters for Q-Learning algorithm:

```python
python -m sandbox.mountain_car__q_learning --search_hyperparams --strict
```

Did not find any feasable configuration accomplishing "strict" requirements.

2. Run Q-Learning algorithm:

```python
python -m sandbox.mountain_car__q_learning --verbose
```

b. Deep Q-Learning algorithm

1. Search hyperparameters for Deep Q-Learning algorithm:

```python
python -m sandbox.mountain_car__deep_q_learning --search_hyperparams --strict
```

Found best configuration:

```python
{
  'alpha': 0.001,
  'gamma': 0.9,
  'c': 16,
  'capacity': 1000,
  'batch_size': 32,
  'reward_speed_coefficient': 100
}
```

2. Run Deep Q-Learning algorithm:

```python
python -m sandbox.mountain_car__deep_q_learning --verbose
```

c. Policy Gradients algorithm

1. Search hyperparameters for Policy Gradients algorithm:

```python
python -m sandbox.mountain_car__policy_gradients_learning --optimize --episodes 50
```

Found best configuration:

```python
{
    'alpha': 0.001758007478624196, 
    'gamma': 0.0021504344145734065, 
    'hidden_size': 16
}
```

## License

[MIT](./LICENSE)
