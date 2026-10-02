# Reinforcement Learning from Scratch to Production

This repository is a hands-on journey through reinforcement learning, covering the foundations, the algorithmic derivations, and the modern production patterns used for LLM alignment and reasoning tuning.

The notebooks are organized in a progression from basic RL theory to applied RL for language models.

## Notebook roadmap

1. [01_foundations_mdps.ipynb](notebooks/01_foundations_mdps.ipynb)  
   Foundations of Markov decision processes, returns, rewards, and the core RL setup.

2. [02_dynamic_programming.ipynb](notebooks/02_dynamic_programming.ipynb)  
   Dynamic programming: policy iteration and value iteration.

3. [03_monte_carlo_td_learning.ipynb](notebooks/03_monte_carlo_td_learning.ipynb)  
   Monte Carlo methods and temporal-difference learning.

4. [04_function_approximation_dqn.ipynb](notebooks/04_function_approximation_dqn.ipynb)  
   Function approximation, deep Q-learning, and replay-based training.

5. [05_policy_gradients.ipynb](notebooks/05_policy_gradients.ipynb)  
   Policy gradients, REINFORCE, baselines, and advantage estimation.

6. [06_trust_regions_ppo.ipynb](notebooks/06_trust_regions_ppo.ipynb)  
   Trust-region methods, importance sampling, KL constraints, TRPO, and PPO.

7. [07_rlhf.ipynb](notebooks/07_rlhf.ipynb)  
   RLHF for language models: reward modeling, preference learning, and PPO-style alignment.

8. [08_rloo_rlvr.ipynb](notebooks/08_rloo_rlvr.ipynb)  
   RLOO and critic-free RL methods, including RLVR-style optimization.

9. [09_grpo_deepseek_r1.ipynb](notebooks/09_grpo_deepseek_r1.ipynb)  
   GRPO and the DeepSeek-style reasoning optimization line of work.

10. [10_dpo_production.ipynb](notebooks/10_dpo_production.ipynb)  
   Direct preference optimization, production PyTorch patterns, and practical `transformers` / `trl` workflows.

11. [11_rlcd.ipynb](notebooks/11_rlcd.ipynb)  
   RLCD: reinforcement learning from contrastive distillation and self-generated preference data.

12. [12_reasoning_slm_grpo.ipynb](notebooks/12_reasoning_slm_grpo.ipynb)  
   A compact demonstration of taking a small non-reasoning SLM and improving it with GRPO and a custom PyTorch training loop.

## How the RL evolves

This course starts from the fundamentals:

- MDPs and the Bellman equations
- value iteration and policy iteration
- Monte Carlo and TD learning
- tabular methods and simple function approximation

It then moves into modern policy optimization:

- REINFORCE and advantage-based methods
- PPO and trust-region style objectives
- reward modeling and RLHF for LLMs
- preference optimization via DPO
- group-based objective methods such as RLOO and GRPO

The final notebooks shift from classic RL to reasoning-focused model optimization, where the reward is tied to correctness, preference quality, or verifier-based signals. The progression is meant to show the conceptual bridge from RL theory to real-world post-training for language models.

## Practical focus

The repo is intentionally educational and code-forward. Many notebooks are designed to show:

- the derivation of the algorithm
- a minimal implementation from scratch
- the same idea in a production library like `torch`, `transformers`, or `trl`
- how to evaluate the model before and after training

## Requirements

Most notebooks assume a Python environment with the common ML stack, including:

- Python 3.10+
- PyTorch
- NumPy
- Matplotlib
- `transformers`
- `trl`
- `datasets`
- `accelerate`
- `gymnasium`

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then open the notebooks in VS Code or Jupyter and follow the numbered sequence.

## License

This project is intended for educational and research use.
