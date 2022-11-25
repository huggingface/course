Let's study how to preprocess a dataset for token classification!

Token classification regroups any task that can be framed as labelling each word (or token) in a sentence, like identifying the persons, organizations and locations for instance.

For our example, we will use the Conll dataset, in which we remove columns we won't use and rename the other ones to get to a dataset with just two columns: words and labels.

If you have your own dataset for token classification, just make sure you clean your data to get to the same point, with one column containing words (as list of strings) and another containing labels (as integers spanning from to to your number of labels -1).() Make sure you have your label names stored somewhere - here we get them from the dataset features - () so you are able to map the integers to some real labels when inspecting your data!

Here we are doing named entity recognitions, so ours labels are either O for words that do not belong to any entity, LOC, for location, PER, for person, ORG for organization and MISC for miscellaneous. Each label has two versions: the B- labels indicate a word that begins an entity while the I- labels indicate a word that is inside an entity.

The first step to preprocess our data is to tokenize the words. This is very easily done with a tokenizer, we just have to tell it we have pre-tokenized the data with the flag is_split_into_words.

Then comes the hard part. Since we have added special tokens and each word may have been split into several tokens, our labels won't match the tokens anymore.

This is where the word IDs our fast tokenizer provide come to the rescue. They match each token to the word it belongs to...

...which allows us to map each token to its label. We just have to make sure we change the B- labels to their I- counterparts for tokens that are inside (but not at the beginning) of a word. The special tokens get a label of -100, which is how we tell the Transformer loss functions to ignore them when computing the loss.

The code is then pretty straightforward, we write a function that shifts the labels for tokens that are inside a word (that you can customize) and use it when generating the labels for each token.

Once that function to create our labels is written, we can preprocess the whole dataset using the map function. With the option batched=True, we unleash the speed of out fast tokenizers.

The last problem comes when we need to create a batch. Unless you changed the preprocessing function to apply some fixed padding, we will get sentences of various lengths, which we need to pad to the same length.

The padding needs to be applied to the inputs as well as the labels, since we should have one label per token.

Again, -100 indicates the labels that should be ignored for the loss computation.

This is all done for us by the DataCollatorForTokenClassification, which you can use in PyTorch or TensorFlow. With all of this, you are either ready to send your data and this data collator to the Trainer, or to use the to_tf_dataset method and use the fit method of your model.