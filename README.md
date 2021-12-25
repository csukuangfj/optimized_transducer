## Introduction

Still working in progress.

This project implements the optimization techniques proposed in
[Improving RNN Transducer Modeling for End-to-End Speech Recognition](https://arxiv.org/abs/1909.12415)
to reduce the memory consumption for computing transducer loss.

During the implementation, we use [torchaudio](https://github.com/pytorch/audio) as a reference.
We ensure that the transducer loss computed using this project is identical to the one computed
using torchaudio when given the same inputs.
