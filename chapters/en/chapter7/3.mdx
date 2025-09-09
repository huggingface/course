# Understanding the DeepSeek R1 Paper

This chapter is a crash course paper reading. We will walk through the paper in simple terms, and then we will break down the key concepts and takeaways.

DeepSeek R1 represents a significant advancement in language model training, particularly in developing reasoning capabilities through reinforcement learning. The paper introduces a new reinforcement learning algorithm called Group Relative Policy Optimization (GRPO).

![DeepSeek R1 Overview](https://huggingface.co/reasoning-course/images/resolve/main/grpo/4.png)

In the next chapter, we will build on this knowledge and implement GRPO in practice.

The initial goal of the paper was to explore whether pure reinforcement learning could develop reasoning capabilities without supervised fine-tuning. 

<Tip>

Up until that point, all the popular LLMs required some supervised fine-tuning, which we explored in [chapter 11](/course/chapter11/1).

</Tip>

## The Breakthrough 'Aha' Moment

![The 'Aha Moment'](https://huggingface.co/reasoning-course/images/resolve/main/grpo/9.png)

One of the most remarkable discoveries in R1-Zero's training was the emergence of a phenomenon known as the "Aha Moment." This phenomenon is somewhat similar to how humans experience sudden realizations while problem-solving. Here's how it works:

1. Initial Attempt: The model makes an initial attempt at solving a problem
2. Recognition: It recognizes potential errors or inconsistencies
3. Self-Correction: It adjusts its approach based on this recognition
4. Explanation: It can explain why the new approach is better

This breakthrough resonates with learners and feels like a "Eureka" moment. It demonstrates learning rather than mere memorization, so let's take a moment to imagine what it feels like to have an "Aha" moment.

For example, imagine you're trying to solve a puzzle:
- First try: "This piece should go here based on the color"
- Recognition: "But wait, the shape doesn't quite fit"
- Correction: "Ah, it actually belongs over there"
- Explanation: "Because both the color and shape pattern match in this position"

This ability emerged naturally from RL training, without being explicitly programmed, demonstrating learning rather than mere memorization of a process from the training data.

The easiest way to understand the 'Aha' moment is to see it in action. Let's take a look at an example. In the chat below, we ask the model to solve a problem and the UI shows the model's thought process as it solves the problem.

<iframe
    src="https://reasoning-course-deepseek-ai-deepseek-r1-distill-0f5fad4.hf.space/"
	frameborder="0"
	width="850"
	height="450"
></iframe>

If you want to try Deepseek's R1, you can also check out [Hugging Chat](https://huggingface.co/chat/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B).

## The Training Process

Training R1 was a multi-phase process. Let's break down the phases and the key innovations in each phase.

The final process results in two models:
- DeepSeek-R1-Zero: A model trained purely using reinforcement learning.
- DeepSeek-R1: A model that builds on the foundation of DeepSeek-R1-Zero and adds supervised fine-tuning.

| Feature | DeepSeek-R1-Zero | DeepSeek-R1 |
|---------|------------------|--------------|
| Training Approach | Pure RL | Multi-phase (SFT + RL) |
| Fine-tuning | None | Supervised fine-tuning |
| Reasoning Capability | Emergent | Enhanced |
| AIME Performance | 71.0% | 79.8% |
| Key Characteristics | Strong reasoning but readability issues | Better language consistency and readability |

While DeepSeek-R1-Zero demonstrates the potential of pure reinforcement learning for developing reasoning capabilities, DeepSeek-R1 builds upon this foundation with a more balanced approach that prioritizes both reasoning performance and usability.

The training process involves four phases:

1. Cold Start Phase
2. Reasoning RL Phase
3. Rejection Sampling Phase
4. Diverse RL Phase

Let's break down each phase:

### Cold Start Phase (Quality Foundation)

![Cold Start Phase](https://huggingface.co/reasoning-course/images/resolve/main/grpo/5.png)

This phase is designed to establish a strong foundation for the model's readability and response quality. It uses a small dataset of high-quality samples from R1-Zero to fine-tune the V3-Base model. Starting with the DeepSeek-V3-Base model, the team used thousands of validated, high-quality samples from R1-Zero for supervised fine-tuning. This innovative approach uses a small but high quality dataset to establish strong baseline readability and response quality.

### Reasoning RL Phase (Capability Building)

![Reasoning RL Phase](https://huggingface.co/reasoning-course/images/resolve/main/grpo/6.png)

The Reasoning RL Phase focuses on developing core reasoning capabilities across domains including mathematics, coding, science, and logic. This phase employs rule-based reinforcement learning, with rewards directly tied to solution correctness. 

Crucially, all the tasks in this phase are 'verifiable' so we can check if the model's answer is correct or not. For example, in the case of mathematics, we can check if the model's answer is correct by using a mathematical solver.

What makes this phase particularly innovative is its direct optimization approach that eliminates the need for a separate reward model, streamlining the training process.

### Rejection Sampling Phase (Quality Control)

![Rejection Sampling Phase](https://huggingface.co/reasoning-course/images/resolve/main/grpo/7.png)

During the Rejection Sampling Phase, the model generates samples which are then filtered through a quality control process. DeepSeek-V3 serves as the quality judge, evaluating outputs across a broad scope that extends beyond pure reasoning tasks. The filtered data is then used for supervised fine-tuning. This phase's innovation lies in its ability to combine multiple quality signals to ensure high-standard outputs.

### Diverse RL Phase (Broad Alignment)

![Diverse RL Phase](https://huggingface.co/reasoning-course/images/resolve/main/grpo/8.png)

The final Diverse RL Phase tackles multiple task types using a sophisticated hybrid approach. For deterministic tasks, it employs rule-based rewards, while subjective tasks are evaluated through LLM feedback. This phase aims to achieve human preference alignment through its innovative hybrid reward approach, combining the precision of rule-based systems with the flexibility of language model evaluation.

## The Algorithm: Group Relative Policy Optimization (GRPO)

Now that we have a good understanding of the training process, let's look at the algorithm that was used to train the model.

The authors describe GRPO as a breakthrough in model fine-tuning:

![GRPO Process](https://huggingface.co/reasoning-course/images/resolve/main/grpo/10.png)

GRPO's novelty lies in its capacity to "directly optimize for preference rectification." This signifies a more direct and efficient route to aligning the model with desired outputs, contrasting with traditional Reinforcement Learning algorithms such as PPO. Let's break down how GRPO works through its three main components.

### Group Formation: Creating Multiple Solutions

The first step in GRPO is remarkably intuitive - it's similar to how a student might solve a difficult problem by trying multiple approaches. When given a prompt, the model doesn't just generate one response; instead, it creates multiple attempts at solving the same problem (usually 4, 8, or 16 different attempts).

Imagine you're teaching a model to solve math problems. For a question about counting chickens on a farm, the model might generate several different solutions:
- One solution might break down the problem step by step: first counting total chickens, then subtracting roosters, and finally accounting for non-laying hens
- Another might use a different but equally valid approach
- Some attempts might contain mistakes or less efficient solutions

All these attempts are kept together as a group, much like having multiple students' solutions to compare and learn from.

![Group Formation](https://huggingface.co/reasoning-course/images/resolve/main/grpo/11.jpg)

### Preference Learning: Understanding What Makes a Good Solution

This is where GRPO really shines in its simplicity. Unlike other methods for RLHF that need always require a separate reward model to predict how good a solution might be, GRPO can use any function or model to evaluate the quality of a solution. For example, we could use a length function to reward shorter responses or a mathematical solver to reward accurate mathematical solutions.

The evaluation process looks at various aspects of each solution:
- Is the final answer correct?
- Did the solution follow proper formatting (like using the right XML tags)?
- Does the reasoning match the answer provided?

What makes this approach particularly clever is how it handles the scoring. Instead of just giving absolute scores, GRPO normalizes the rewards within each group. It uses a simple but effective formula for group relative advantage estimation:

```
Advantage = (reward - mean(group_rewards)) / std(group_rewards)
```

![Preference Learning](https://huggingface.co/reasoning-course/images/resolve/main/grpo/12.jpg)

This normalization is like grading on a curve, but for AI. It helps the model understand which solutions within the group were better or worse compared to their peers, rather than just looking at absolute scores.

### Optimization: Learning from Experience

The final step is where GRPO teaches the model to improve based on what it learned from evaluating the group of solutions. This process is both powerful and stable, using two main principles:

1. It encourages the model to produce more solutions like the successful ones while moving away from less effective approaches
2. It includes a safety mechanism (called KL divergence penalty) that prevents the model from changing too drastically all at once

This approach proves more stable than traditional methods because:
- It looks at multiple solutions together rather than comparing just two at a time
- The group-based normalization helps prevent issues with reward scaling
- The KL penalty acts like a safety net, ensuring the model doesn't forget what it already knows while learning new things

<Tip>

GRPO's key innovations are:
- Learning directly from any function or model, eliminating the reliance on a separate reward model.
- Group-based learning, which is more stable and efficient than traditional methods like pairwise comparisons.

</Tip>

This breakdown is complex, but the key takeaway is that GRPO is a more efficient and stable way to train a model to reason. 

### GRPO Algorithm in Pseudocode

Now that we understand the key components of GRPO, let's look at the algorithm in pseudocode. This is a simplified version of the algorithm, but it captures the key ideas.

```
Input: 
- initial_policy: Starting model to be trained
- reward_function: Function that evaluates outputs
- training_prompts: Set of training examples
- group_size: Number of outputs per prompt (typically 4-16)

Algorithm GRPO:
1. For each training iteration:
   a. Set reference_policy = initial_policy (snapshot current policy)
   b. For each prompt in batch:
      i. Generate group_size different outputs using initial_policy
      ii. Compute rewards for each output using reward_function
      iii. Normalize rewards within group:
           normalized_advantage = (reward - mean(rewards)) / std(rewards)
      iv. Update policy by maximizing the clipped ratio:
          min(prob_ratio * normalized_advantage, 
              clip(prob_ratio, 1-epsilon, 1+epsilon) * normalized_advantage)
          - kl_weight * KL(initial_policy || reference_policy)
          
          where prob_ratio is current_prob / reference_prob

Output: Optimized policy model
```

This algorithm shows how GRPO combines group-based advantage estimation with policy optimization while maintaining stability through clipping and KL divergence constraints.

## Results and Impact

Now that we've explored the algorithm, let's look at the results. DeepSeek R1 achieves state-of-the-art performance across multiple domains:

| Domain | Key Results |
|--------|-------------|
| Mathematics | • 79.8% on AIME 2024<br>• 97.3% on MATH-500 |
| Coding | • Codeforces Rating: 2029<br>• LiveCodeBench: 65.9% |
| General Knowledge | • MMLU: 90.8%<br>• GPQA Diamond: 71.5% |
| Language Tasks | • AlpacaEval 2.0: 87.6% win rate<br>• FRAMES: 82.5% |

The model's practical impact extends beyond benchmarks through its cost-effective API pricing ($0.14 per million input tokens) and successful model distillation across various sizes (1.5B to 70B parameters). Notably, even the 7B model achieves 55.5% on AIME 2024, while the 70B distilled version approaches o1-mini performance on MATH-500 (94.5%), demonstrating effective capability preservation at different scales.

## Limitations and Challenges of GRPO

While GRPO represents a significant advancement in reinforcement learning for language models, it's important to understand its limitations and challenges:

- **Generation Cost**: Generating multiple completions (4-16) for each prompt increases computational requirements compared to methods that generate only one or two completions.
- **Batch Size Constraints**: The need to process groups of completions together can limit effective batch sizes, adding complexity to the training process and potentially slowing down training.
- **Reward Function Design**: The quality of training heavily depends on well-designed reward functions. Poorly designed rewards can lead to unintended behaviors or optimization for the wrong objectives.
- **Group Size Tradeoffs**: Choosing the optimal group size involves balancing diversity of solutions against computational cost. Too few samples may not provide enough diversity, while too many increase training time and resource requirements.
- **KL Divergence Tuning**: Finding the right balance for the KL divergence penalty requires careful tuning - too high and the model won't learn effectively, too low and it may diverge too far from its initial capabilities.

## Conclusion

The DeepSeek R1 paper represents a significant milestone in language model development. The Group Relative Policy Optimization (GRPO) algorithm has demonstrated that pure reinforcement learning can indeed develop strong reasoning capabilities, challenging previous assumptions about the necessity of supervised fine-tuning.

Perhaps most importantly, DeepSeek R1 has shown that it's possible to balance high performance with practical considerations like cost-effectiveness and accessibility. The successful distillation of the model's capabilities across different sizes, from 1.5B to 70B parameters, demonstrates a path forward for making advanced AI capabilities more widely available.

---

In the next section, we'll explore practical implementations of these concepts, focusing on how to leverage GRPO and RFTrans in your own language model development projects.

## Quiz

### 1. What is the main innovation of the DeepSeek R1 paper?

<Question
    choices={[
        {
            text: "The GRPO algorithm that enables learning from preferences with and without a reward model",
            explain: "Correct! GRPO's key innovation is its ability to directly optimize for preference rectification, making it more efficient than traditional RL methods.",
            correct: true
        },
        {
            text: "Using more GPUs for training than any previous model",
            explain: "The paper's innovation is in its algorithmic approach (GRPO) rather than computational resources used."
        },
        {
            text: "Creating a larger language model than existing ones",
            explain: "The innovation lies in the training methodology and GRPO algorithm, not in model size."
        }
    ]}
/>

### 2. What are the four phases of the DeepSeek R1 training process?

<Question
    choices={[
        {
            text: "Cold Start, Reasoning RL, Rejection Sampling, and Diverse RL",
            explain: "Correct! These are the exact four phases described in the paper, each serving a specific purpose in the training pipeline.",
            correct: true
        },
        {
            text: "Pre-training, Fine-tuning, Testing, and Deployment",
            explain: "The paper specifically outlines four different phases: Cold Start (quality foundation), Reasoning RL (capability building), Rejection Sampling (quality control), and Diverse RL (broad alignment)."
        },
        {
            text: "Data Collection, Model Training, Evaluation, and Optimization",
            explain: "The paper describes a specific four-phase process that includes Cold Start, Reasoning RL, Rejection Sampling, and Diverse RL phases."
        }
    ]}
/>

### 3. What is the 'Aha Moment' phenomenon in R1-Zero's training?

<Question
    choices={[
        {
            text: "A process where the model recognizes errors, self-corrects, and explains its corrections",
            explain: "Correct! The paper describes this as a four-step process: initial attempt, recognition of errors, self-correction, and explanation of the improvement.",
            correct: true
        },
        {
            text: "The point where the model reaches human-level performance",
            explain: "The 'Aha Moment' specifically refers to the model's ability to recognize and correct its own mistakes, similar to human problem-solving realizations."
        },
        {
            text: "When the model completes its training process",
            explain: "The 'Aha Moment' is about the model's emergent ability to recognize errors and self-correct during problem-solving, not about training completion."
        }
    ]}
/>

### 4. How does GRPO's group formation work?

<Question
    choices={[
        {
            text: "It generates multiple solutions (4-16) for the same problem and evaluates them together",
            explain: "Correct! GRPO generates multiple attempts at solving the same problem, typically 4, 8, or 16 different attempts, which are then evaluated as a group.",
            correct: true
        },
        {
            text: "It combines multiple models into one ensemble",
            explain: "GRPO uses a single model to generate multiple solution attempts, not an ensemble of different models."
        },
        {
            text: "It splits the training data into different groups",
            explain: "GRPO's group formation involves generating multiple solutions for the same problem, not splitting training data."
        }
    ]}
/>

### 5. What is the key difference between DeepSeek-R1-Zero and DeepSeek-R1?

<Question
    choices={[
        {
            text: "R1-Zero uses pure RL while R1 combines RL with supervised fine-tuning",
            explain: "Correct! As shown in the comparison table, R1-Zero uses pure RL training while R1 uses a multi-phase approach combining supervised fine-tuning with RL, resulting in better language consistency.",
            correct: true
        },
        {
            text: "R1-Zero is smaller than R1",
            explain: "The difference is in their training approaches (pure RL vs. multi-phase), not their model sizes."
        },
        {
            text: "R1-Zero was trained on less data",
            explain: "The key distinction is their training methodology: pure RL for R1-Zero versus a combined SFT and RL approach for R1."
        }
    ]}
/>

