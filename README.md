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

### Part III — Agentic RL

13. [13_agentic_rl_foundations.ipynb](notebooks/13_agentic_rl_foundations.ipynb)  
   What an LLM agent is, a topic map of agentic RL, the history from prompting to imitation to RL, and the POMDP formalism. Experiments: behavior cloning vs DAgger vs RL as the horizon grows, why RL needs an SFT prior, and curricula.

14. [14_agent_loop_multiturn_rl.ipynb](notebooks/14_agent_loop_multiturn_rl.ipynb)  
   The agent loop as an RL environment: a text tool environment, a batched multi-turn rollout engine, SFT warm start and multi-turn GRPO with observation masking. Ablations: training without the mask, and tool-call costs.

15. [15_credit_assignment_multiturn.ipynb](notebooks/15_credit_assignment_multiturn.ipynb)  
   Credit assignment across turns: trajectory-level GRPO vs step-level grouping (GiGPO) vs a turn-level critic with GAE, discounting, and process rewards (potential-based shaping vs a reward that gets hacked).

16. [16_agentic_rl_slm_tools.ipynb](notebooks/16_agentic_rl_slm_tools.ipynb)  
   Training a real tool-using agent: Qwen2.5-0.5B-Instruct + a calculator, with multi-turn GRPO in `trl` (`tools=`, `environment_factory=`) and from scratch in PyTorch.

17. [17_agentic_rl_at_scale.ipynb](notebooks/17_agentic_rl_at_scale.ipynb)  
   Environment hacking (an agent that edits the tests) and defenses, async rollouts and off-policy corrections (truncated IS, decoupled PPO, sequence-length effects), non-verifiable rewards, instabilities, frameworks, benchmarks, and a recipe for agentic RL.

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

Part III takes the next step, from models that *reason* to agents that *act*. The model calls tools and reads their output over many turns, and is trained from outcome rewards. It covers why RL (rather than imitation) is the method of choice for agents, multi-turn rollouts with observation masking, credit assignment across turns, a real tool-using SLM, and the reward-hacking and systems issues that appear at scale.

## Practical focus

The repo is intentionally educational and code-forward. Many notebooks are designed to show:

- the derivation of the algorithm
- a minimal implementation from scratch
- the same idea in a production library like `torch`, `transformers`, or `trl`
- how to evaluate the model before and after training

## Interactive explainers

- [Forward vs. reverse KL](docs/kl-divergence/index.html): interactively explore mode covering, mode seeking, and approximating a known true posterior.
- [The spring inside KL divergence](docs/kl-spring/index.html): see how a KL penalty behaves like a spring near a reference policy.

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
