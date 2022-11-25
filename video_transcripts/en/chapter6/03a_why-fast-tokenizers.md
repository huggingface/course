Why are fast tokenizers called fast? In this video we will see exactly how much faster the so-called fast tokenizers are compared to their slow counterparts.

For this benchmark, we will use the GLUE MNLI dataset, which contains 432 thousands pairs of texts. We will see how long it takes for the fast and slow versions of a BERT tokenizer to process them all.

We define our fast and slow tokenizer using the AutoTokenizer API. The fast tokenizer is the default (when available), so we pass along use_fast=False to define the slow one.

In a notebook, we can time the execution of a cell with the time magic command, like this. Processing the whole dataset is four times faster with a fast tokenizer. That's quicker indeed, but not very impressive however.

That's because we passed along the texts to the tokenizer one at a time. This is a common mistake to do with fast tokenizers, which are backed by Rust and thus able to parallelize the tokenization of multiple texts. Passing them only one text at a time is like sending a cargo ship between two continents with just one container, it's very inefficient. To unleash the full speed of our fast tokenizers, we need to send them batches of texts, which we can do with the batched=True argument of the map method.

Now those results are impressive! The fast tokenizer takes 12 seconds to process a dataset that takes 4 minutes to the slow tokenizer.

Summarizing the results in this table, you can see why we have called those tokenizers fast. And this is only for tokenizing texts. If you ever need to train a new tokenizer, they do this very quickly too!