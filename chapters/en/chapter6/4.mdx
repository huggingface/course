<CourseFloatingBanner chapter={2}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/main/course/en/chapter11/section4.ipynb"},
]} />

# LoRA (Low-Rank Adaptation)

Fine-tuning large language models is a resource intensive process. LoRA is a technique that allows us to fine-tune large language models with a small number of parameters. It works by adding and optimizing smaller matrices to the attention weights, typically reducing trainable parameters by about 90%.

## Understanding LoRA

LoRA (Low-Rank Adaptation) is a parameter-efficient fine-tuning technique that freezes the pre-trained model weights and injects trainable rank decomposition matrices into the model's layers. Instead of training all model parameters during fine-tuning, LoRA decomposes the weight updates into smaller matrices through low-rank decomposition, significantly reducing the number of trainable parameters while maintaining model performance. For example, when applied to GPT-3 175B, LoRA reduced trainable parameters by 10,000x and GPU memory requirements by 3x compared to full fine-tuning. You can read more about LoRA in the [LoRA paper](https://arxiv.org/pdf/2106.09685).

LoRA works by adding pairs of rank decomposition matrices to transformer layers, typically focusing on attention weights. During inference, these adapter weights can be merged with the base model, resulting in no additional latency overhead. LoRA is particularly useful for adapting large language models to specific tasks or domains while keeping resource requirements manageable.

## Key advantages of LoRA

1. **Memory Efficiency**: 
   - Only adapter parameters are stored in GPU memory
   - Base model weights remain frozen and can be loaded in lower precision
   - Enables fine-tuning of large models on consumer GPUs

2. **Training Features**:
   - Native PEFT/LoRA integration with minimal setup
   - Support for QLoRA (Quantized LoRA) for even better memory efficiency

3. **Adapter Management**:
   - Adapter weight saving during checkpoints
   - Features to merge adapters back into base model

## Loading LoRA Adapters with PEFT

[PEFT](https://github.com/huggingface/peft) is a library that provides a unified interface for loading and managing PEFT methods, including LoRA. It allows you to easily load and switch between different PEFT methods, making it easier to experiment with different fine-tuning techniques. 

Adapters can be loaded onto a pretrained model with `load_adapter()`, which is useful for trying out different adapters whose weights aren't merged. Set the active adapter weights with the `set_adapter()` function. To return the base model, you could use unload() to unload all of the LoRA modules. This makes it easy to switch between different task-specific weights.  

```python
from peft import PeftModel, PeftConfig

config = PeftConfig.from_pretrained("ybelkada/opt-350m-lora")
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(model, "ybelkada/opt-350m-lora")
```

![lora_load_adapter](https://github.com/huggingface/smol-course/raw/main/3_parameter_efficient_finetuning/images/lora_adapter.png)

## Fine-tune LLM using `trl` and the `SFTTrainer` with LoRA

The [SFTTrainer](https://huggingface.co/docs/trl/sft_trainer) from `trl` provides integration with LoRA adapters through the [PEFT](https://huggingface.co/docs/peft/en/index) library. This means that we can fine-tune a model in the same way as we did with SFT, but use LoRA to reduce the number of parameters we need to train.

We'll use the `LoRAConfig` class from PEFT in our example. The setup requires just a few configuration steps:

1. Define the LoRA configuration (rank, alpha, dropout)
2. Create the SFTTrainer with PEFT config
3. Train and save the adapter weights

## LoRA Configuration

Let's walk through the LoRA configuration and key parameters.

| Parameter | Description |
|-----------|-------------|
| `r` (rank) | Dimension of the low-rank matrices used for weight updates. Typically between 4-32. Lower values provide more compression but potentially less expressiveness. |
| `lora_alpha` | Scaling factor for LoRA layers, usually set to 2x the rank value. Higher values result in stronger adaptation effects. |
| `lora_dropout` | Dropout probability for LoRA layers, typically 0.05-0.1. Higher values help prevent overfitting during training. |
| `bias` | Controls training of bias terms. Options are "none", "all", or "lora_only". "none" is most common for memory efficiency. |
| `target_modules` | Specifies which model modules to apply LoRA to. Can be "all-linear" or specific modules like "q_proj,v_proj". More modules enable greater adaptability but increase memory usage. |

<Tip>
When implementing PEFT methods, start with small rank values (4-8) for LoRA and monitor training loss. Use validation sets to prevent overfitting and compare results with full fine-tuning baselines when possible. The effectiveness of different methods can vary by task, so experimentation is key.
</Tip>

## Using TRL with PEFT

PEFT methods can be combined with TRL for fine-tuning to reduce memory requirements. We can pass the  `LoraConfig` to the model when loading it.  

```python
from peft import LoraConfig

# TODO: Configure LoRA parameters
# r: rank dimension for LoRA update matrices (smaller = more compression)
rank_dimension = 6
# lora_alpha: scaling factor for LoRA layers (higher = stronger adaptation)
lora_alpha = 8
# lora_dropout: dropout probability for LoRA layers (helps prevent overfitting)
lora_dropout = 0.05

peft_config = LoraConfig(
    r=rank_dimension,  # Rank dimension - typically between 4-32
    lora_alpha=lora_alpha,  # LoRA scaling factor - typically 2x rank
    lora_dropout=lora_dropout,  # Dropout probability for LoRA layers
    bias="none",  # Bias type for LoRA. the corresponding biases will be updated during training.
    target_modules="all-linear",  # Which modules to apply LoRA to
    task_type="CAUSAL_LM",  # Task type for model architecture
)
```

Above, we used `device_map="auto"` to automatically assign the model to the correct device. You can also manually assign the model to a specific device using `device_map={"": device_index}`. 

We will also need to define the `SFTTrainer` with the LoRA configuration.

```python
# Create SFTTrainer with LoRA configuration
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    peft_config=peft_config,  # LoRA configuration
    max_seq_length=max_seq_length,  # Maximum sequence length
    processing_class=tokenizer,
)
```

<Tip>

✏️ **Try it out!** Build on your fine-tuned model from the previous section, but fine-tune it with LoRA. Use the `HuggingFaceTB/smoltalk` dataset to fine-tune a `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model, using the LoRA configuration we defined above.

</Tip>

## Merging LoRA Adapters

After training with LoRA, you might want to merge the adapter weights back into the base model for easier deployment. This creates a single model with the combined weights, eliminating the need to load adapters separately during inference.

The merging process requires attention to memory management and precision. Since you'll need to load both the base model and adapter weights simultaneously, ensure sufficient GPU/CPU memory is available. Using `device_map="auto"` in `transformers` will find the correct device for the model based on your hardware. 

Maintain consistent precision (e.g., float16) throughout the process, matching the precision used during training and saving the merged model in the same format for deployment. 

## Merging Implementation

After training a LoRA adapter, you can merge the adapter weights back into the base model. Here's how to do it:

```python
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel

# 1. Load the base model
base_model = AutoModelForCausalLM.from_pretrained(
    "base_model_name", torch_dtype=torch.float16, device_map="auto"
)

# 2. Load the PEFT model with adapter
peft_model = PeftModel.from_pretrained(
    base_model, "path/to/adapter", torch_dtype=torch.float16
)

# 3. Merge adapter weights with base model
merged_model = peft_model.merge_and_unload()
```

If you encounter size discrepancies in the saved model, ensure you're also saving the tokenizer:

```python
# Save both model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("base_model_name")
merged_model.save_pretrained("path/to/save/merged_model")
tokenizer.save_pretrained("path/to/save/merged_model")
```

<Tip>

✏️ **Try it out!** Merge the adapter weights back into the base model. Use the `HuggingFaceTB/smoltalk` dataset to fine-tune a `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` model, using the LoRA configuration we defined above.

</Tip>


# Resources

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Hugging Face blog post on PEFT](https://huggingface.co/blog/peft)
