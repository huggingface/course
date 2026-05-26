# Evaluation

With a finetuned model through either SFT or LoRA SFT, we should evaluate it on standard benchmarks. As machine learning engineers you should maintain a suite of relevant evaluations for your targeted domain of interest. In this page, we will look at some of the most common benchmarks and how to use them to evaluate your model. We'll also look at how to create custom benchmarks for your specific use case.

## Automatic Benchmarks

Automatic benchmarks serve as standardized tools for evaluating language models across different tasks and capabilities. While they provide a useful starting point for understanding model performance, it's important to recognize that they represent only one piece of a comprehensive evaluation strategy.

## Understanding Automatic Benchmarks

Automatic benchmarks typically consist of curated datasets with predefined tasks and evaluation metrics. These benchmarks aim to assess various aspects of model capability, from basic language understanding to complex reasoning. The key advantage of using automatic benchmarks is their standardization - they allow for consistent comparison across different models and provide reproducible results.

However, it's crucial to understand that benchmark performance doesn't always translate directly to real-world effectiveness. A model that excels at academic benchmarks may still struggle with specific domain applications or practical use cases.

## General Knowledge Benchmarks

[MMLU](https://huggingface.co/datasets/cais/mmlu) (Massive Multitask Language Understanding) tests knowledge across 57 subjects, from science to humanities. While comprehensive, it may not reflect the depth of expertise needed for specific domains. TruthfulQA evaluates a model's tendency to reproduce common misconceptions, though it can't capture all forms of misinformation.

## Reasoning Benchmarks

[BBH](https://huggingface.co/datasets/lukaemon/bbh) (Big Bench Hard) and [GSM8K](https://huggingface.co/datasets/openai/gsm8k) focus on complex reasoning tasks. BBH tests logical thinking and planning, while GSM8K specifically targets mathematical problem-solving. These benchmarks help assess analytical capabilities but may not capture the nuanced reasoning required in real-world scenarios.

## Language Understanding

[HELM](https://github.com/stanford-crfm/helm) provides a holistic evaluation framework. Benchmarks like HELM offer insights into language processing capabilities on aspects like commonsense, world knowledge, and reasoning. But may not fully represent the complexity of natural conversation or domain-specific terminology.

## Domain-Specific Benchmarks

Let's look at a few benchmarks that focus on specific domains like math, coding, and chat.

The [MATH benchmark](https://huggingface.co/papers/2103.03874) is another important evaluation tool for mathematical reasoning. It consists of 12,500 problems from mathematics competitions, covering algebra, geometry, number theory, counting, probability, and more. What makes MATH particularly challenging is that it requires multi-step reasoning, formal mathematical notation understanding, and the ability to generate step-by-step solutions. Unlike simpler arithmetic tasks, MATH problems often demand sophisticated problem-solving strategies and mathematical concept applications.

The [HumanEval Benchmark](https://github.com/openai/human-eval) is a coding-focused evaluation dataset consisting of 164 programming problems. The benchmark tests a model's ability to generate functionally correct Python code that solves the given programming tasks. What makes HumanEval particularly valuable is that it evaluates both code generation capabilities and functional correctness through actual test case execution, rather than just superficial similarity to reference solutions. The problems range from basic string manipulation to more complex algorithms and data structures.

[Alpaca Eval](https://tatsu-lab.github.io/alpaca_eval/) is an automated evaluation framework designed to assess the quality of instruction-following language models. It uses GPT-4 as a judge to evaluate model outputs across various dimensions including helpfulness, honesty, and harmlessness. The framework includes a dataset of 805 carefully curated prompts and can evaluate responses against multiple reference models like Claude, GPT-4, and others. What makes Alpaca Eval particularly useful is its ability to provide consistent, scalable evaluations without requiring human annotators, while still capturing nuanced aspects of model performance that traditional metrics might miss.

## Alternative Evaluation Approaches

Many organizations have developed alternative evaluation methods to address the limitations of standard benchmarks:

### LLM-as-Judge

Using one language model to evaluate another's outputs has become increasingly popular. This approach can provide more nuanced feedback than traditional metrics, though it comes with its own biases and limitations.

### Evaluation Arenas

Evaluation arenas like [Chatbot Arena](https://lmarena.ai/) offer a unique approach to LLM assessment through crowdsourced feedback. In these platforms, users engage in anonymous "battles" between two LLMs, asking questions and voting on which model provides better responses. This approach captures real-world usage patterns and preferences through diverse, challenging questions, with studies showing strong agreement between crowd-sourced votes and expert evaluations. While powerful, these platforms have limitations including potential user base bias, skewed prompt distributions, and a primary focus on helpfulness rather than safety considerations.

### Custom Benchmark Suites

Organizations often develop internal benchmark suites tailored to their specific needs and use cases. These might include domain-specific knowledge tests or evaluation scenarios that mirror actual deployment conditions.

## Custom Evaluation

While standard benchmarks provide a useful baseline, they shouldn't be your only evaluation method. Here's how to develop a more comprehensive approach:

1. Start with relevant standard benchmarks to establish a baseline and enable comparison with other models.

2. Identify the specific requirements and challenges of your use case. What tasks will your model actually perform? What kinds of errors would be most problematic?

3. Develop custom evaluation datasets that reflect your actual use case. This might include:
   - Real user queries from your domain
   - Common edge cases you've encountered
   - Examples of particularly challenging scenarios

4. Consider implementing a multi-layered evaluation strategy:
   - Automated metrics for quick feedback
   - Human evaluation for nuanced understanding
   - Domain expert review for specialized applications
   - A/B testing in controlled environments

## Implementing Custom Evaluations

In this section, we will implement evaluation for our finetuned model. We can use [`lighteval`](https://github.com/huggingface/lighteval) to evaluate our finetuned model on standard benchmarks, which contains a wide range of tasks built into the library. We just need to define the tasks we want to evaluate and the parameters for the evaluation.  

LightEval tasks are defined using a specific format:

```
{suite}|{task}|{num_few_shot}|{auto_reduce}
```

| Parameter | Description |
|-----------|-------------|
| `suite` | The benchmark suite (e.g., 'mmlu', 'truthfulqa') |
| `task` | Specific task within the suite (e.g., 'abstract_algebra') |
| `num_few_shot` | Number of examples to include in prompt (0 for zero-shot) |
| `auto_reduce` | Whether to automatically reduce few-shot examples if prompt is too long (0 or 1) |

Example: `"mmlu|abstract_algebra|0|0"` evaluates on MMLU's abstract algebra task with zero-shot inference.

## Example Evaluation Pipeline

Let's set up an evaluation pipeline for our finetuned model. We will evaluate the model on  set of sub tasks that relate to the domain of medicine. 

Here's a complete example of evaluating on automatic benchmarks relevant to one specific domain using Lighteval with the VLLM backend:

```bash
lighteval accelerate \
    "pretrained=your-model-name" \
    "mmlu|anatomy|0|0" \
    "mmlu|high_school_biology|0|0" \
    "mmlu|high_school_chemistry|0|0" \
    "mmlu|professional_medicine|0|0" \
    --max_samples 40 \
    --batch_size 1 \
    --output_path "./results" \
    --save_generations true
```

Results are displayed in a tabular format showing:

```
|                  Task                  |Version|Metric|Value |   |Stderr|
|----------------------------------------|------:|------|-----:|---|-----:|
|all                                     |       |acc   |0.3333|±  |0.1169|
|leaderboard:mmlu:_average:5             |       |acc   |0.3400|±  |0.1121|
|leaderboard:mmlu:anatomy:5              |      0|acc   |0.4500|±  |0.1141|
|leaderboard:mmlu:high_school_biology:5  |      0|acc   |0.1500|±  |0.0819|
```

Lighteval also include a python API for more detailed evaluation tasks, which is useful for manipulating the results in a more flexible way. Check out the [Lighteval documentation](https://huggingface.co/docs/lighteval/using-the-python-api) for more information.

> [!TIP]
> ✏️ **Try it out!** Evaluate your finetuned model on a specific task in lighteval.

# End-of-chapter quiz[[end-of-chapter-quiz]]

<CourseFloatingBanner
    chapter={11}
    classNames="absolute z-10 right-0 top-0"
/>

### 1. What are the main advantages of using automatic benchmarks for model evaluation?

<Question
	choices={[
		{
			text: "They provide perfect real-world performance metrics",
			explain: "Incorrect! While automatic benchmarks are useful, they don't always translate directly to real-world performance."
		},
		{
			text: "They allow for standardized comparison between models and provide reproducible results",
			explain: "Correct! This is one of the key benefits of automatic benchmarks.",
			correct: true
		},
		{
			text: "They eliminate the need for any other form of evaluation",
			explain: "Incorrect! Automatic benchmarks should be part of a comprehensive evaluation strategy, not the only method."
		}
	]}
/>

### 2. Which benchmark specifically tests knowledge across 57 different subjects?

<Question
	choices={[
		{
			text: "BBH (Big Bench Hard)",
			explain: "Incorrect! BBH focuses on complex reasoning tasks, not broad subject knowledge."
		},
		{
			text: "GSM8K",
			explain: "Incorrect! GSM8K specifically targets mathematical problem-solving."
		},
		{
			text: "MMLU",
			explain: "Correct! MMLU (Massive Multitask Language Understanding) tests knowledge across 57 subjects, from science to humanities.",
			correct: true
		}
	]}
/>

### 3. What is LLM-as-Judge?

<Question
	choices={[
		{
			text: "Using one language model to evaluate another's outputs",
			explain: "Correct! This is an alternative evaluation approach that can provide more nuanced feedback.",
			correct: true
		},
		{
			text: "A benchmark that tests judicial reasoning",
			explain: "Incorrect! LLM-as-Judge refers to using one model to evaluate another, not testing judicial reasoning."
		},
		{
			text: "A method for training models on legal datasets",
			explain: "Incorrect! This isn't related to training on legal data, but rather using one model to evaluate another's outputs."
		}
	]}
/>

### 4. What should be included in a comprehensive evaluation strategy?

<Question
	choices={[
		{
			text: "Only standard benchmarks",
			explain: "Incorrect! A comprehensive strategy should include multiple evaluation methods."
		},
		{
			text: "Standard benchmarks, custom evaluation datasets, and domain-specific testing",
			explain: "Correct! A comprehensive strategy should include multiple layers of evaluation.",
			correct: true
		},
		{
			text: "Only custom datasets specific to your use case",
			explain: "Incorrect! While custom datasets are important, they shouldn't be the only evaluation method."
		}
	]}
/>

### 5. What is a limitation of automatic benchmarks?

<Question
	choices={[
		{
			text: "They are too expensive to run",
			explain: "Incorrect! Cost isn't typically the main limitation of automatic benchmarks."
		},
		{
			text: "Benchmark performance doesn't always translate directly to real-world effectiveness",
			explain: "Correct! This is a key limitation to keep in mind when using automatic benchmarks.",
			correct: true
		},
		{
			text: "They can only evaluate small models",
			explain: "Incorrect! Automatic benchmarks can be used to evaluate models of various sizes."
		}
	]}
/>

### 6. What is the purpose of creating custom evaluation datasets?

<Question
	choices={[
		{
			text: "To reflect your specific use case and include real user queries from your domain",
			explain: "Correct! Custom datasets help ensure evaluation is relevant to your specific needs.",
			correct: true
		},
		{
			text: "To replace standard benchmarks entirely",
			explain: "Incorrect! Custom datasets should complement, not replace, standard benchmarks."
		},
		{
			text: "To make evaluation easier",
			explain: "Incorrect! Creating custom datasets requires additional effort but provides more relevant evaluation."
		}
	]}
/>
