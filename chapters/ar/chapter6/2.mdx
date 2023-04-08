# Training a new tokenizer from an old one[[training-a-new-tokenizer-from-an-old-one]]

<CourseFloatingBanner chapter={6}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter6/section2.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter6/section2.ipynb"},
]} />

If a language model is not available in the language you are interested in, or if your corpus is very different from the one your language model was trained on, you will most likely want to retrain the model from scratch using a tokenizer adapted to your data. That will require training a new tokenizer on your dataset. But what exactly does that mean? When we first looked at tokenizers in [Chapter 2](/course/chapter2), we saw that most Transformer models use a _subword tokenization algorithm_. To identify which subwords are of interest and occur most frequently in the corpus at hand, the tokenizer needs to take a hard look at all the texts in the corpus -- a process we call *training*. The exact rules that govern this training depend on the type of tokenizer used, and we'll go over the three main algorithms later in this chapter.

<Youtube id="DJimQynXZsQ"/>

<Tip warning={true}>

⚠️ Training a tokenizer is not the same as training a model! Model training uses stochastic gradient descent to make the loss a little bit smaller for each batch. It's randomized by nature (meaning you have to set some seeds to get the same results when doing the same training twice). Training a tokenizer is a statistical process that tries to identify which subwords are the best to pick for a given corpus, and the exact rules used to pick them depend on the tokenization algorithm. It's deterministic, meaning you always get the same results when training with the same algorithm on the same corpus.

</Tip>

## Assembling a corpus[[assembling-a-corpus]]

There's a very simple API in 🤗 Transformers that you can use to train a new tokenizer with the same characteristics as an existing one: `AutoTokenizer.train_new_from_iterator()`. To see this in action, let’s say we want to train GPT-2 from scratch, but in a language other than English. Our first task will be to gather lots of data in that language in a training corpus. To provide examples everyone will be able to understand, we won't use a language like Russian or Chinese here, but rather a specialized English language: Python code.

The [🤗 Datasets](https://github.com/huggingface/datasets) library can help us assemble a corpus of Python source code. We'll use the usual `load_dataset()` function to download and cache the [CodeSearchNet](https://huggingface.co/datasets/code_search_net) dataset. This dataset was created for the [CodeSearchNet challenge](https://wandb.ai/github/CodeSearchNet/benchmark) and contains millions of functions from open source libraries on GitHub in several programming languages. Here, we will load the Python part of this dataset:

```py
from datasets import load_dataset

# This can take a few minutes to load, so grab a coffee or tea while you wait!
raw_datasets = load_dataset("code_search_net", "python")
```

We can have a look at the training split to see which columns we have access to:

```py
raw_datasets["train"]
```

```python out
Dataset({
    features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 
      'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 
      'func_code_url'
    ],
    num_rows: 412178
})
```

We can see the dataset separates docstrings from code and suggests a tokenization of both. Here. we'll just use the `whole_func_string` column to train our tokenizer. We can look at an example of one these functions by indexing into the `train` split:

```py
print(raw_datasets["train"][123456]["whole_func_string"])
```

which should print the following:

```out
def handle_simple_responses(
      self, timeout_ms=None, info_cb=DEFAULT_MESSAGE_CALLBACK):
    """Accepts normal responses from the device.

    Args:
      timeout_ms: Timeout in milliseconds to wait for each response.
      info_cb: Optional callback for text sent from the bootloader.

    Returns:
      OKAY packet's message.
    """
    return self._accept_responses('OKAY', info_cb, timeout_ms=timeout_ms)
```

The first thing we need to do is transform the dataset into an _iterator_ of lists of texts -- for instance, a list of list of texts. Using lists of texts will enable our tokenizer to go faster (training on batches of texts instead of processing individual texts one by one), and it should be an iterator if we want to avoid having everything in memory at once. If your corpus is huge, you will want to take advantage of the fact that 🤗 Datasets does not load everything into RAM but stores the elements of the dataset on disk. 

Doing the following would create a list of lists of 1,000 texts each, but would load everything in memory:

```py
# Don't uncomment the following line unless your dataset is small!
# training_corpus = [raw_datasets["train"][i: i + 1000]["whole_func_string"] for i in range(0, len(raw_datasets["train"]), 1000)]
```

Using a Python generator, we can avoid Python loading anything into memory until it's actually necessary. To create such a generator, you just to need to replace the brackets with parentheses:

```py
training_corpus = (
    raw_datasets["train"][i : i + 1000]["whole_func_string"]
    for i in range(0, len(raw_datasets["train"]), 1000)
)
```

This line of code doesn't fetch any elements of the dataset; it just creates an object you can use in a Python `for` loop. The texts will only be loaded when you need them (that is, when you're at the step of the `for` loop that requires them), and only 1,000 texts at a time will be loaded. This way you won't exhaust all your memory even if you are processing a huge dataset.

The problem with a generator object is that it can only be used once. So, instead of this giving us the list of the first 10 digits twice:

```py
gen = (i for i in range(10))
print(list(gen))
print(list(gen))
```

we get them once and then an empty list:

```python out
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
[]
```

That's why we define a function that returns a generator instead:

```py
def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 1000]["whole_func_string"]
        for i in range(0, len(raw_datasets["train"]), 1000)
    )


training_corpus = get_training_corpus()
```

You can also define your generator inside a `for` loop by using the `yield` statement:

```py
def get_training_corpus():
    dataset = raw_datasets["train"]
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx : start_idx + 1000]
        yield samples["whole_func_string"]
```

which will produce the exact same generator as before, but allows you to use more complex logic than you can in a list comprehension.

## Training a new tokenizer[[training-a-new-tokenizer]]

Now that we have our corpus in the form of an iterator of batches of texts, we are ready to train a new tokenizer. To do this, we first need to load the tokenizer we want to pair with our model (here, GPT-2):

```py
from transformers import AutoTokenizer

old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
```

Even though we are going to train a new tokenizer, it's a good idea to do this to avoid starting entirely from scratch. This way, we won't have to specify anything about the tokenization algorithm or the special tokens we want to use; our new tokenizer will be exactly the same as GPT-2, and the only thing that will change is the vocabulary, which will be determined by the training on our corpus.

First let's have a look at how this tokenizer would treat an example function:

```py
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

tokens = old_tokenizer.tokenize(example)
tokens
```

```python out
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo',
 'Ġnumbers', 'Ġ`', 'a', '`', 'Ġand', 'Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```

This tokenizer has a few special symbols, like `Ġ` and `Ċ`, which denote spaces and newlines, respectively. As we can see, this is not too efficient: the tokenizer returns individual tokens for each space, when it could group together indentation levels (since having sets of four or eight spaces is going to be very common in code). It also split the function name a bit weirdly, not being used to seeing words with the `_` character.

Let's train a new tokenizer and see if it solves those issues. For this, we'll use the method `train_new_from_iterator()`:

```py
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
```

This command might take a bit of time if your corpus is very large, but for this dataset of 1.6 GB of texts it's  blazing fast (1 minute 16 seconds on an AMD Ryzen 9 3900X CPU with 12 cores).

Note that `AutoTokenizer.train_new_from_iterator()` only works if the tokenizer you are using is a "fast" tokenizer. As you'll see in the next section, the 🤗 Transformers library contains two types of tokenizers: some are written purely in Python and others (the fast ones) are backed by the 🤗 Tokenizers library, which is written in the [Rust](https://www.rust-lang.org) programming language. Python is the language most often used for data science and deep learning applications, but when anything needs to be parallelized to be fast, it has to be written in another language. For instance, the matrix multiplications that are at the core of the model computation are written in CUDA, an optimized C library for GPUs.

Training a brand new tokenizer in pure Python would be excruciatingly slow, which is why we developed the 🤗 Tokenizers library. Note that just as you didn't have to learn the CUDA language to be able to execute your model on a batch of inputs on a GPU, you won't need to learn Rust to use a fast tokenizer. The 🤗 Tokenizers library provides Python bindings for many methods that internally call some piece of code in Rust; for example, to parallelize the training of your new tokenizer or, as we saw in [Chapter 3](/course/chapter3), the tokenization of a batch of inputs.

Most of the Transformer models have a fast tokenizer available (there are some exceptions that you can check [here](https://huggingface.co/transformers/#supported-frameworks)), and the `AutoTokenizer` API always selects the fast tokenizer for you if it's available. In the next section we'll take a look at some of the other special features fast tokenizers have, which will be really useful for tasks like token classification and question answering. Before diving into that, however, let's try our brand new tokenizer on the previous example:

```py
tokens = tokenizer.tokenize(example)
tokens
```

```python out
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"""', 'Add', 'Ġthe', 'Ġtwo', 'Ġnumbers', 'Ġ`',
 'a', '`', 'Ġand', 'Ġ`', 'b', '`."""', 'ĊĠĠĠ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
```

Here we again see the special symbols `Ġ` and `Ċ` that denote spaces and newlines, but we can also see that our tokenizer learned some tokens that are highly specific to a corpus of Python functions: for example, there is a `ĊĠĠĠ` token that represents an indentation, and a `Ġ"""` token that represents the three quotes that start a docstring. The tokenizer also correctly split the function name on `_`. This is quite a compact representation; comparatively, using the plain English tokenizer on the same example will give us a longer sentence:

```py
print(len(tokens))
print(len(old_tokenizer.tokenize(example)))
```

```python out
27
36
```

Let's look at another example:

```python
example = """class LinearLayer():
    def __init__(self, input_size, output_size):
        self.weight = torch.randn(input_size, output_size)
        self.bias = torch.zeros(output_size)

    def __call__(self, x):
        return x @ self.weights + self.bias
    """
tokenizer.tokenize(example)
```

```python out
['class', 'ĠLinear', 'Layer', '():', 'ĊĠĠĠ', 'Ġdef', 'Ġ__', 'init', '__(', 'self', ',', 'Ġinput', '_', 'size', ',',
 'Ġoutput', '_', 'size', '):', 'ĊĠĠĠĠĠĠĠ', 'Ġself', '.', 'weight', 'Ġ=', 'Ġtorch', '.', 'randn', '(', 'input', '_',
 'size', ',', 'Ġoutput', '_', 'size', ')', 'ĊĠĠĠĠĠĠĠ', 'Ġself', '.', 'bias', 'Ġ=', 'Ġtorch', '.', 'zeros', '(',
 'output', '_', 'size', ')', 'ĊĊĠĠĠ', 'Ġdef', 'Ġ__', 'call', '__(', 'self', ',', 'Ġx', '):', 'ĊĠĠĠĠĠĠĠ',
 'Ġreturn', 'Ġx', 'Ġ@', 'Ġself', '.', 'weights', 'Ġ+', 'Ġself', '.', 'bias', 'ĊĠĠĠĠ']
```

In addition to the token corresponding to an indentation, here we can also see a token for a double indentation: `ĊĠĠĠĠĠĠĠ`. The special Python words like `class`, `init`, `call`, `self`, and `return` are each tokenized as one token, and we can see that as well as splitting on `_` and `.` the tokenizer correctly splits even camel-cased names: `LinearLayer` is tokenized as `["ĠLinear", "Layer"]`.

## Saving the tokenizer[[saving-the-tokenizer]]

To make sure we can use it later, we need to save our new tokenizer. Like for models, this is done with the `save_pretrained()` method:

```py
tokenizer.save_pretrained("code-search-net-tokenizer")
```

This will create a new folder named *code-search-net-tokenizer*, which will contain all the files the tokenizer needs to be reloaded. If you want to share this tokenizer with your colleagues and friends, you can upload it to the Hub by logging into your account. If you're working in a notebook, there's a convenience function to help you with this:

```python
from huggingface_hub import notebook_login

notebook_login()
```

This will display a widget where you can enter your Hugging Face login credentials. If you aren't working in a notebook, just type the following line in your terminal:

```bash
huggingface-cli login
```

Once you've logged in, you can push your tokenizer by executing the following command:

```py
tokenizer.push_to_hub("code-search-net-tokenizer")
```

This will create a new repository in your namespace with the name `code-search-net-tokenizer`, containing the tokenizer file. You can then load the tokenizer from anywhere with the `from_pretrained()` method:

```py
# Replace "huggingface-course" below with your actual namespace to use your own tokenizer
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")
```

You're now all set for training a language model from scratch and fine-tuning it on your task at hand! We'll get to that in [Chapter 7](/course/chapter7), but first, in the rest of this chapter we'll take a closer look at fast tokenizers and explore in detail what actually happens when we call the method `train_new_from_iterator()`.
