Let's see how to preprocess a dataset for translation.

This is the task of... well translating a sentence in another language.

This video will focus on how to preprocess your dataset once you have managed to put it in the following format: one column for the input texts, and one for the target texts. Here is how we can achieve this with the Datasets library on the KDE4 dataset for English to French translation.

As long as you manage to have your data look like this, you should be able to follow the same steps.

For once, our labels are not integers corresponding to some classes, but plain text. We will thus need to tokenize them, like our inputs. There is a trap there though, as if you tokenize your targets like your inputs, you will hit a problem. Even if you don't speak French, you might notice some weird things in the tokenization of the targets: most of the words are tokenized in several subtokens, while "fish", one of the only English word, is tokenized as a single word.

That's because our inputs have been tokenized as English. Since our model knows two languages, you have to warn it when tokenizing the targets, so it swtiches in French mode. This is done with the as_target_tokenizer context manager. You can see how it results in a more compact tokenization.

Processing the whole dataset is then super easy with the map function. You can pick different maximum lengths for the input and targets, and choose to pad at this stage to that maximum length by setting paddin=max_length. Here we will show you how to pad dynamically as it requires one more step.

Your inputs and targets are all sentence of various lengths.

We will pad the inputs and targets separately as the maximum length of the inputs and targets might be different.

Then we pad the inputs with the pad token and the targets with the -100 index, to make sure they are not taken into account in the loss computation.

Once this is done, batching inputs and targets become super easy!

The Transformers library provides us with a data collator to do this all automatically. You can then pass it to the Trainer with your datasets, or use it in the to_tf_dataset method before using model.fit().