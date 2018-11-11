# MoonBoard

This repo is an experimental for utilizing Machine Learning technology for Rock Climbing training board named *MoonBoard*.

## Directory organization

```
datasets/                # Dataset from MoonBoard
  +- create_moonboard.py # Create moonboard.npz from climbs.txt/grades.txt
  +- climbs.txt          # Dataset from michaelplesser/moonboard-NN
  +- grades.txt          # Dataset from michaelplesser/moonboard-NN
  +- load_moonboard.py   # Data loader for MoonBoard
  +- moonboard.npz       # Training and Test datasets for MoonBoard
```

## Usage

### Loading dataset from Python

- Its usage is quite similar to `keras.datasets.mnist.load_data()` function.
- The argument represents the path to `moonboard.npz` to be loaded.
- The return values are same as `keras.datasets.mnist.load_data()`.

```python
>>> from datasets.load_moonboard import load_moonboard
>>> (x_train, y_train), (x_test, y_test) = load_moonboard('./datasets/moonboard.npz')
>>> print(x_train[0])
  [[0 0 0 0 0 0 0 1 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 1 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 1 0 0 0 0 0 0 0 1]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 1 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 1 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]
  [0 0 0 0 0 0 0 0 0 0 0]]
>>> print(y_train[0])
  10
```

### Using with TensorFlow

- TBA

```python
```

## Contribution

## References

