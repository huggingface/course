In this video we will see together what is the purpose of training a tokenizer, what are the key steps to follow and what is the easiest way to do it.

You will ask yourself the question "Should I train a new tokenizer?" when you plan to train a new model from scratch.

A trained tokenizer would not be suitable for your corpus if your corpus is in a different language, uses new characters - such as accents or upper cased letters - , has a specific vocabulary - for example medical or legal - or uses a different style - a language from another century for instance.

For example, if I take the tokenizer trained for the bert-base-uncased model and ignore its normalization step then we can see that the tokenization operation on the English sentence "here is a sentence adapted to our tokenizer" produces a rather satisfactory list of tokens in the sense that this sentence of 8 words is tokenized into 9 tokens.

On the other hand if we use this same tokenizer on a sentence in Bengali, we see that either a word is divided into many sub tokens or that the tokenizer does not know one of the unicode characters and returns only an unknown token. The fact that a "common" word is split into many tokens can be problematic because language models can only handle a sequence of tokens of limited length. A tokenizer that excessively splits your initial text may even impact the performance of your model. Unknown tokens are also problematic because the model will not be able to extract any information from the "unknown" part of the text.

In this other example, we can see that the tokenizer replaces words containing characters with accents and capital letters with unknown tokens. Finally, if we use again this tokenizer to tokenize medical vocabulary we see again that a single word is divided into many sub tokens: 4 for "paracetamol" and "pharyngitis".



Most of the tokenizers used by the current state of the art language models need to be trained on a  corpus that is similar to the one used to pre-train the language model. This training consists in learning rules to divide the text into tokens and the way to learn these rules and use them depends on the chosen tokenizer model

Thus, to train a new tokenizer it is first necessary to build a training corpus composed of raw texts. Then, you have to choose an architecture for your tokenizer. Here there are two options: the simplest is to reuse the same architecture as the one of a tokenizer used by another model already trained,otherwise it is also possible to completely design your tokenizer but it requires more experience and attention. Once the architecture is chosen, one can thus train this tokenizer on your constituted corpus. Finally, the last thing that you need to do is to save the learned rules to be able to use this tokenizer which is now ready to be used.

Let's take an example: let's say you want to train a GPT-2 model on Python code. Even if the python code is in English this type of text is very specific and deserves a tokenizer trained on it - to convince you of this we will see at the end the difference produced on an example. For that we are going to use the method "train_new_from_iterator" that all the fast tokenizers of the library have and thus in particular GPT2TokenizerFast. This is the simplest method in our case to have a tokenizer adapted to python code.

Remember, the first step is to gather a training corpus. We will use a subpart of the CodeSearchNet dataset containing only python functions from open source libraries on Github. It's good timing, this dataset is known by the datasets library and we can load it in two lines of code. Then, as the "train_new_from_iterator" method expects a iterator of lists of texts we create the "get_training_corpus" function which will return an iterator.

Now that we have our iterator on our python functions corpus, we can load the gpt-2 tokenizer architecture. Here "old_tokenizer" is not adapted to our corpus but we only need one more line to train it on our new corpus. An argument that is common to most of the tokenization algorithms used at the moment is the size of the vocabulary, we choose here the value 52 thousand. Finally, once the training is finished, we just have to save our new tokenizer locally or send it to the hub to be able to reuse it very easily afterwards.

Finally, let's see together on an example if it was useful to re-train a tokenizer similar to gpt2 one. With the original tokenizer of GPT-2 we see that all spaces are isolated and the method name "randn" relatively common in python code is split in 2. With our new tokenizer, single and double indentations have been learned and the method "randn" is tokenized into 1 token. And with that, you now know how to train your very own tokenizers!