# Random Speculative Sampling Algorithm for Accelerating Large Language Model Decoding
implementation speculative sampling
@ Itai Shapira

This repository contains an implementation of the Random Speculative Sampling Algorithm proposed in the paper "Accelerating Large Language Model Decoding with Speculative Sampling." (https://arxiv.org/pdf/2302.01318.pdf).
This repository was created as a pset solution for Harvard CS229.

## Introduction
The transformer architecture has a significant advantage over earlier sequence models in terms of parallel processing. However, when generating new sequences with transformers during inference time, each output token is dependent on all previously generated tokens, making the models run *serially* for each token generated. This can be a slow and computationally expensive process, especially for large models with billions of parameters. Additionally, the inability to perform batched inference wastes computational resources as large model inference is typically limited by memory bandwidth rather than compute power. To address this issue, the authors of the paper above propose leveraging the fact that certain tokens are easier to predict than others and can be generated accurately with smaller and weaker but faster models. This technique is not limited to transformers and can be applied to any large autoregressive language model.


## Paper Overview
The paper proposes a technique called Speculative Sampling to speed up decoding in large language models. Decoding can be slow and computationally expensive in large models, as each token prediction requires running the model forward through multiple layers. Speculative Sampling reduces this computational cost by parallelizing the decoding process and speculatively predicting multiple tokens at once. This sampling scheme preserves the distribution of the target model.

## Algorithm Description
The algorithm uses two models: small-but-fast model (in this case, gpt2) and large-but-slow model (in this case gpt2-large).

Given a t-token prefix, the algorithm generates k possible tokens sequentially using the slow-but-fast model. Next, using the big model, we compute the distrubtions of next-tokens in parallel using the provisional tokens of the small model.
Next, we’ll perform a kind of rejection sampling to combine our sets of predictions, in a way that presevers the orginial distrubtion of the big model:
Sample $r \sim U(0, 1)^k$. Iterating over $t < i \leq t + k$, compute $\frac{p_b(x_{s_i} \mid x_{<i})}{p_s(x_{s_i} \mid x_{<i})}$. 

If $r_i$ is greater than this quotient, 
record the index $i = i^\star$ and break. 


## Repository Overview
This repository contains an implementation of the Random Speculative Sampling Algorithm in Python using the PyTorch library. 

## Remarks
For maximal speedups, the small model should be at least an order of magnitude smaller than the large one. Yet, since the vocabularies of the two models need to be the same, we're stuck with gpt-2. It should be possible to fit both models on a single Colab GPU.

The implementation also contains autoregressive runtimes for the small model and the large model, and compares those to runtimes for the efficient attention algorithm.

If you are having trouble observing a speedup, use an extremely "predictable" prompt where the large model and the small model agree, like "A B C D". This will make it easier for the efficient inference algorithm to skip executions of the large model.


