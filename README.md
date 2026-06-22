# Reinforcement Learning from Human Feedback (RLHF) for Language Models

## _This project is self-implemented_

The goal of this project is to implement a complete RLHF pipeline for language models, including supervised fine-tuning (SFT), reward model training, and reinforcement learning (RL) with Proximal Policy Optimization (PPO).
This is based on tutorials from Ashwani Kumar et al. [tutorial](https://www.youtube.com/watch?v=K1UBOodkqEk)


## I. Experiments

### A. Supervised Fine-Tuning (SFT) on SST-2 dataset
* Model: GPT-2
* Dataset: SST-2
* Batch size: 32

1. Exp.1
    + Epochs: 3
    + Learning rate: 3e-5 ---> 0. (Cosine Annealing LR)

    **---> Validation Loss: 4.055375133241926 after training for 3 epoch(s)**

2. Exp.2
    + Epochs: 1
    + Learning rate: 3e-5 ---> 1e-5 (Cosine Annealing LR)

    **---> Validation Loss: 3.8983952403068542 after training for 1 epoch(s)**

3. Exp.3
    + Epochs: 1
    + Learning rate: 5e-5 ---> 1e-5 (Cosine Annealing LR)

    **---> Validation Loss: 3.9061887860298157 after training for 1 epoch(s)**
