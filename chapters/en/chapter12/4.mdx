# Implementing GRPO in TRL

In this page, we'll learn how to implement Group Relative Policy Optimization (GRPO) using the Transformer Reinforcement Learning (TRL) library. We'll focus on practical implementation with minimal code.

We'll explore the core concepts of GRPO as they are embodied in TRL's GRPOTrainer, using snippets from the official TRL documentation to guide us.

<Tip>

This chapter is aimed at TRL beginners. If you are already familiar with TRL, you might want to also check out the [Open R1 implementation](https://github.com/huggingface/open-r1/blob/main/src/open_r1/grpo.py) of GRPO.

</Tip>

First, let's remind ourselves of some of the important concepts of GRPO algorithm:

- Group Formation: The model generates multiple completions for each prompt.
- Preference Learning: The model learns from a reward function that compares groups of completions.
- Training Configuration: The model uses a configuration to control the training process.

What do we need to do to implement GRPO?

- Define a dataset of prompts.
- Define a reward function that takes a list of completions and returns a list of rewards.
- Configure the training process with a GRPOConfig.
- Train the model using the GRPOTrainer.

Here's a minimal example to get started with GRPO training:

```python
from trl import GRPOTrainer, GRPOConfig
from datasets import load_dataset

# 1. Load your dataset
dataset = load_dataset("your_dataset", split="train")


# 2. Define a simple reward function
def reward_func(completions, **kwargs):
    """Example: Reward longer completions"""
    return [float(len(completion)) for completion in completions]


# 3. Configure training
training_args = GRPOConfig(
    output_dir="output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    logging_steps=10,
)

# 4. Initialize and train
trainer = GRPOTrainer(
    model="your_model",  # e.g. "Qwen/Qwen2-0.5B-Instruct"
    args=training_args,
    train_dataset=dataset,
    reward_funcs=reward_func,
)
trainer.train()
```

## Key Components

### 1. Dataset Format

Your dataset should contain prompts that the model will respond to. The GRPO trainer will generate multiple completions for each prompt and use the reward function to compare them.

### 2. Reward Function

The reward function is crucial - it determines how the model learns. Here are two practical examples:

```python
# Example 1: Reward based on completion length
def reward_length(completions, **kwargs):
    return [float(len(completion)) for completion in completions]


# Example 2: Reward based on matching a pattern
import re


def reward_format(completions, **kwargs):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    return [1.0 if re.match(pattern, c) else 0.0 for c in completions]
```

### 3. Training Configuration

Key parameters to consider in `GRPOConfig`:

```python
training_args = GRPOConfig(
    # Essential parameters
    output_dir="output",
    num_train_epochs=3,
    num_generation=4,  # Number of completions to generate for each prompt
    per_device_train_batch_size=4,  # We want to get all generations in one device batch
    # Optional but useful
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    logging_steps=10,
    # GRPO specific (optional)
    use_vllm=True,  # Speed up generation
)
```

The `num_generation` parameter is particularly important for GRPO as it defines the group size - how many different completions the model will generate for each prompt. This is a key differentiator from other RL methods:

- Too small (e.g., 2-3): May not provide enough diversity for meaningful comparisons
- Recommended (4-16): Provides good balance between diversity and computational efficiency
- Larger values: May improve learning but significantly increases computational cost

The group size should be chosen based on your computational resources and the complexity of your task. For simple tasks, smaller groups (4-8) may be sufficient, while more complex reasoning tasks might benefit from larger groups (8-16).

## Tips for Success

1. **Memory Management**: Adjust `per_device_train_batch_size` and `gradient_accumulation_steps` based on your GPU memory.
2. **Speed**: Enable `use_vllm=True` for faster generation if your model is supported.
3. **Monitoring**: Watch the logged metrics during training:
   - `reward`: Average reward across completions
   - `reward_std`: Standard deviation within reward groups
   - `kl`: KL divergence from reference model

## Reward Function Design

The DeepSeek R1 paper demonstrates several effective approaches to reward function design that you can adapt for your own GRPO implementation:

### 1. Length-Based Rewards

One of the easiest rewards to implement is a length-based reward. You can reward longer completions:

```python
def reward_len(completions, **kwargs):
    ideal_length = 20
    return [-abs(ideal_length - len(completion)) for completion in completions]
```

This reward function penalizes completions that are too short or too long, encouraging the model to generate completions that are close to the ideal length of 20 tokens.

<!-- # TODO: update links when PR is merged -->

<iframe 
    src="https://marimo.app/gh/huggingface/notebooks/main/e?entrypoint=course%2Fen%2Fchapter13%2Fgrpo_length.py&embed=true&show-chrome=false"
    title="Marimo Notebook"
    width="100%"
    height="800px"
    frameBorder="0"
    allow="clipboard-write"
></iframe>

## 2. Rule-Based Rewards for Verifiable Tasks

For tasks with objectively correct answers (like mathematics or coding), you can implement rule-based reward functions:

```python
def problem_reward(completions, answers, **kwargs):
    """Reward function for math problems with verifiable answers
    completions: list of completions to evaluate
    answers: list of answers to the problems from the dataset
    """

    rewards = []
    for completion, correct_answer in zip(completions, answers):
        # Extract the answer from the completion
        try:
            # This is a simplified example - you'd need proper parsing
            answer = extract_final_answer(completion)
            # Binary reward: 1 for correct, 0 for incorrect
            reward = 1.0 if answer == correct_answer else 0.0
            rewards.append(reward)
        except:
            # If we can't parse an answer, give a low reward
            rewards.append(0.0)

    return rewards
```

<!-- # TODO: update links when PR is merged -->

<iframe 
    src="https://marimo.app/gh/huggingface/notebooks/main/e?entrypoint=course%2Fen%2Fchapter13%2Fgrpo_math.py&embed=true&show-chrome=false"
    title="Marimo Notebook"
    width="100%"
    height="800px"
    frameBorder="0"
    allow="clipboard-write"
></iframe>

## 3. Format-Based Rewards

You can also reward proper formatting, which was important in the DeepSeek R1 training:

```python
def format_reward(completions, **kwargs):
    """Reward completions that follow the desired format"""
    # Example: Check if the completion follows a think-then-answer format
    pattern = r"<think>(.*?)</think>\s*<answer>(.*?)</answer>"

    rewards = []
    for completion in completions:
        match = re.search(pattern, completion, re.DOTALL)
        if match:
            # Check if there's substantial content in both sections
            think_content = match.group(1).strip()
            answer_content = match.group(2).strip()

            if len(think_content) > 20 and len(answer_content) > 0:
                rewards.append(1.0)
            else:
                rewards.append(
                    0.5
                )  # Partial reward for correct format but limited content
        else:
            rewards.append(0.0)  # No reward for incorrect format

    return rewards
```

<!-- # TODO: update links when PR is merged -->

<iframe 
    src="https://marimo.app/gh/huggingface/notebooks/main/e?entrypoint=course%2Fen%2Fchapter13%2Fgrpo_format.py&embed=true&show-chrome=false"
    title="Marimo Notebook"
    width="100%"
    height="800px"
    frameBorder="0"
    allow="clipboard-write"
></iframe>



These examples demonstrate how you can implement reward functions inspired by the DeepSeek R1 training process, focusing on correctness, formatting, and combined signals.


## That's it!

In the next section, you will follow an exercise to implement GRPO in TRL.

