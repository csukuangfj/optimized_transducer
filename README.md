## Introduction

This project implements the optimization techniques proposed in
[Improving RNN Transducer Modeling for End-to-End Speech Recognition](https://arxiv.org/abs/1909.12415)
to reduce the memory consumption for computing transducer loss.

During the implementation, we use [torchaudio](https://github.com/pytorch/audio) as a reference.
We ensure that the transducer loss computed using this project is identical to the one computed
using torchaudio when giving the same inputs.

## Installation

You can install it via `pip`:

```
pip install optimized_transducer
```

### Installation FAQ

### What operating systems are supported ?

It has been tested on Ubuntu 18.04. It should also work on macOS and other unixes systems.
It may work on Windows, though it is not tested.

### How to display installation log ?

Use

```
pip install --verbose optimized_transducer
```

### How to reduce installation time ?

Use

```
export OT_MAKE_ARGS="-j"
pip install --verbose optimized_transducer
```

### Which version of PyTorch is supported ?

It has been tested on PyTorch >= 1.5.0. It may work on PyTorch < 1.5.0


### How to install a CPU version of `optimized_transducer` ?

Use

```
export OT_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DOT_WITH_CUDA=OFF"
pip install --verbose optimized_transducer
```
