# Transformer Translation Model from Scratch

## Overview

This project aims to build a translation model leveraging the Transformer architecture from scratch. The Transformer model, introduced in the paper ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al., has become a cornerstone in the field of natural language processing, providing state-of-the-art results in tasks such as translation, summarization, and more.

The architecture is mainly composed of two parts: the encoder and the decoder, both of which have multiple layers of self-attention and feed-forward neural networks.

![Transformer Architecture](https://machinelearningmastery.com/wp-content/uploads/2021/10/transformer_1.png)



## Getting Started

### Prerequisites

- Python 3.9 or newer
- PyTorch 1.7 or newer
- MLflow (for experiment tracking)

### Installation

Clone the repository and install the required packages using the following commands:

```bash
git clone https://github.com/iftesha1/Transformer_from_scratch_Pytorch.git
cd transformer_from_scratch
poetry shell
poetry install


