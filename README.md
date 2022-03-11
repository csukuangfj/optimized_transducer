## Introduction

This project implements the optimization techniques proposed in
[Improving RNN Transducer Modeling for End-to-End Speech Recognition](https://arxiv.org/abs/1909.12415)
to reduce the memory consumption for computing transducer loss.

**HINT**: You can find ASR training code using this repo
in <https://github.com/k2-fsa/icefall>. You can also find
decoding code in [icefall](https://github.com/k2-fsa/icefall).

### How does it differ from the RNN-T loss from torchaudio


It produces same output as [torchaudio](https://github.com/pytorch/audio)
for the same input, so `optimized_transducer` should be equivalent to
[torchaudio.functional.rnnt_loss()](https://github.com/pytorch/audio/blob/main/torchaudio/functional/functional.py#L1546).

This project is more memory efficient (See <https://github.com/csukuangfj/transducer-loss-benchmarking>
for benchmark results)

Also, `torchaudio` accepts only output from `nn.Linear`, but
we also support output from `log-softmax` (You can set the option
`from_log_softmax` to `True` in this case).

It also supports a **modified** version of transducer. See [below](#modified-transducer) for what
the meaning of **modified transducer** is.

### How does it differ from [warp-transducer](https://github.com/HawkAaron/warp-transducer)

It borrows the methods of computing alpha and beta from `warp-transducer`. Therefore,
`optimized_transducer` produces the same `alpha` and `beta` as `warp-transducer`
for the same input.


However, `warp-transducer` produces different gradients for CPU and CUDA
when using the same input. See <https://github.com/HawkAaron/warp-transducer/issues/93>.
I also created a [colab notebook](https://colab.research.google.com/drive/1vMkH8LmiCCOiCo4KTTEcv-NU8_OGn0ie?usp=sharing)
to reproduce that issue.

This project produces consistent gradient on CPU and CUDA for the same input, just like
what `torchaudio` is doing. (We borrow the gradient computation formula from `torchaudio`).

`optimized_transducer` uses less memory than that of `warp-transducer`
(See <https://github.com/csukuangfj/transducer-loss-benchmarking> for benchmark results).

It also supports a **modified** version of transducer. See [below](#modified-transducer) for what
the meaning of **modified transducer** is.

### Modified Transducer

In **modified transducer**, we limit the maximum number of symbols per frame to 1. The following
figure compares the formula for forward and backward procedures between standard transducer
and modified transducer.

<img src="/pic.svg" width="514" height="507.5" />

**Note**: Modified transducer is proposed independently by [@danpovey](https://github.com/danpovey).
We were later informed that the idea already existed in
[Recurrent Neural Aligner: An Encoder-Decoder Neural Network Model for
Sequence to Sequence Mapping](https://www.isca-speech.org/archive_v0/Interspeech_2017/pdfs/1705.PDF)


## Installation

You can install it via `pip`:

```
pip install optimized_transducer
```

To check that `optimized_transducer` was installed successfully, please run

```
python3 -c "import optimized_transducer; print(optimized_transducer.__version__)"
```

which should print the version of the installed `optimized_transducer`, e.g., `1.2`.

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

It will pass `-j` to `make`.

### Which version of PyTorch is supported ?

It has been tested on PyTorch >= 1.5.0. It may work on PyTorch < 1.5.0


### How to install a CPU version of `optimized_transducer` ?

Use

```
export OT_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DOT_WITH_CUDA=OFF"
export OT_MAKE_ARGS="-j"
pip install --verbose optimized_transducer
```

It will pass `-DCMAKE_BUILD_TYPE=Release -DOT_WITH_CUDA=OFF` to `cmake`.

### What Python versions are supported ?

Python >= 3.6 is known to work. It may work for Python 2.7, though it is not tested.

### Where to get help if I have problems with the installation ?

Please file an issue at <https://github.com/csukuangfj/optimized_transducer/issues>
and describe your problem there.

## Usage

`optimized_transducer` expects that the output shape of the joint network is
**NOT** `(N, T, U, V)`, but is `(sum_all_TU, V)`, which is a concatenation
of 2-D tensors: `(T_1 * U_1, V)`, `(T_2 * U_2, V)`, ..., `(T_N, U_N, V)`.
**Note**: `(T_1 * U_1, V)` is just the reshape of a 3-D tensor `(T_1, U_1, V)`.


Suppose your original joint network looks somewhat like the following:

```python3
encoder_out = torch.rand(N, T, D) # from the encoder
decoder_out = torch.rand(N, U, D) # from the decoder, i.e., the prediction network

encoder_out = encoder_out.unsqueeze(2) # Now encoder out is (N, T, 1, D)
decoder_out = decoder_out.unsqueeze(1) # Now decoder out is (N, 1, U, D)

x = encoder_out + decoder_out # x is of shape (N, T, U, D)
activation = torch.tanh(x)

logits = linear(activation) # linear is an instance of `nn.Linear`.

loss = torchaudio.functional.rnnt_loss(
    logits=logits,
    targets=targets,
    logit_lengths=logit_lengths,
    target_lengths=target_lengths,
    blank=blank_id,
    reduction="mean",
)
```

You need to change it to the following:

```python3
encoder_out = torch.rand(N, T, D) # from the encoder
decoder_out = torch.rand(N, U, D) # from the decoder, i.e., the prediction network

encoder_out_list = [encoder_out[i, :logit_lengths[i], :] for i in range(N)]
decoder_out_list = [decoder_out[i, :target_lengths[i]+1, :] for i in range(N)]

x = [e.unsqueeze(1) + d.unsqueeze(0) for e, d in zip(encoder_out_list, decoder_out_list)]
x = [p.reshape(-1, D) for p in x]
x = torch.cat(x)

activation = torch.tanh(x)
logits = linear(activation) # linear is an instance of `nn.Linear`.

loss = optimized_transducer.transducer_loss(
    logits=logits,
    targets=targets,
    logit_lengths=logit_lengths,
    target_lengths=target_lengths,
    blank=blank_id,
    reduction="mean",
    from_log_softmax=False,
)
```

**Caution**: We used `from_log_softmax=False` in the above example since `logits`
is the output of `nn.Linear`.

**Hint**: If `logits` is the output of `log-softmax`, you should use `from_log_softmax=True`.

In most cases, you should pass the output of `nn.Linear` to compute the loss, i.e.,
use `from_log_softmax=False`, to save memory.

If you want to do some operations on the output of `log-softmax` before feeding it
to `optimized_transducer.transducer_loss()`, `from_log_softmax=True` is helpful in
this case. But be aware that this will increase the memory usage.


To use the **modified** transducer, pass an additional argument `one_sym_per_frame=True`
to `optimized_transducer.transducer_loss()`.

For more usages, please refer to

  - <https://github.com/csukuangfj/optimized_transducer/blob/master/optimized_transducer/python/optimized_transducer/transducer_loss.py>
  - <https://github.com/csukuangfj/optimized_transducer/blob/master/optimized_transducer/python/tests/test_cuda.py>
  - <https://github.com/csukuangfj/optimized_transducer/blob/master/optimized_transducer/python/tests/test_compute_transducer_loss.py>
  - <https://github.com/csukuangfj/optimized_transducer/blob/master/optimized_transducer/python/tests/test_max_symbol_per_frame.py>

## For developers

As a developer, you don't need to use `pip install optimized_transducer`.
To make development easier, you can use

```
git clone https://github.com/csukuangfj/optimized_transducer.git
cd optimized_transducer
mkdir build
cd build
cmake -DOT_BUILD_TESTS=ON -DCMAKE_BUILD_TYPE=Release ..
make -j
export PYTHONPATH=$PWD/../optimized_transducer/python:$PWD/lib:$PYTHONPATH
```

I usually create a file `path.sh` inside the `build` directory, containing

```
export PYTHONPATH=$PWD/../optimized_transducer/python:$PWD/lib:$PYTHONPATH
```

so what you need to do is
```
cd optimized_transducer/build
source path.sh

# Then you are ready to run Python tests
python3 optimized_transducer/python/tests/test_compute_transducer_loss.py

# You can also use "import optimized_transducer" in your Python projects
```

To run all Python tests, use

```
cd optimized_transducer/build
ctest --output-on-failure
```

Alternatively one can "make" all available tests

```
make -j test
```
