Let's study how to preprocess a dataset for question answering!

Question answering is the task of finding answers to a question in some context.

For our example, we will use the squad dataset, in which we remove columns we won't use and just extract the information we will need for the labels: the start and the end of the answer in the context.

If you have your own dataset for question answering, just make sure you clean your data to get to the same point, with one column containing the questions, one column containing the contexts, one column for the index of the start and end character of the answer in the context.

Note that the answer must be part of the context. If you want to perform generative question answering, look at one of the sequence to sequence videos linked below.

Now if we have a look at the tokens we will feed our model...

... we will see the answer lies somewhere inside the context.

For very long context that answer may get truncated

by the tokenizer. In this case, we wont have any proper labels for our model.

So we should keep the truncated part as a separate feature instead of discarding it.

The only thing we need to be careful with, is to allow some overlap between separate chunks so that the answer is not truncated, and that the feature containing the answer gets sufficient context to be able to predict it.

Here is how it can be done by the tokenizer: we pass it the question, context, set the truncation for the context only and the padding to the maximum length. The stride argument is where we set the number of overlapping tokens, and the return_overflowing_tokens means we don't want to discard the truncated part. Lastly, we also return the offset mappings to be able to find the tokens corresponding to the answer start and end.

We want those two tokens, because there will be the labels we pass to our model. In a one-hot encoded version, here is what they look like.

If the context we have does not contain the answer, we set the two labels to the index of the CLS token.

We also do this if the context only partially contains the answer.

In terms of code, here is how we can do it: using the sequence IDs of an input, we can determine the beginning and the end of the context. Then we know if have to return the CLS position for the two labels or we determine the positions of the first and last tokens of the answer.

We can check it works properly on our previous example.

Putting it all together looks like this big function, which

we can apply to our datasets. Since we applied padding during the tokenization, we can then use this directly in the Trainer or apply the to_tf_dataset method to use Keras.fit.