<FrameworkSwitchCourse {fw} />

# Debugging the training pipeline

<DocNotebookDropdown
  classNames="absolute z-10 right-0 top-0"
  options={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/chapter8/section4_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/chapter8/section4_tf.ipynb"},
]} />

You've written a beautiful script to train or fine-tune a model on a given task, dutifully following the advice from [Chapter 7](/course/chapter7). But when you launch the command `model.fit()`, something horrible happens: you get an error 😱! Or worse, everything seems to be fine and the training runs without error, but the resulting model is crappy. In this section, we will show you what you can do to debug these kinds of issues.

## Debugging the training pipeline

<Youtube id="N9kO52itd0Q"/>

The problem when you encounter an error in `model.fit()` is that it could come from multiple sources, as training usually brings together a lot of things that you've been working on up until that point. The problem could be something wrong in your dataset, or some issue when trying to batch elements of the datasets together. Or it could be something wrong in the model code, or your loss function or optimizer. And even if everything goes well for training, something could still go wrong during the evaluation if there is a problem with your metric.

The best way to debug an error that arises in `model.fit()` is to manually go through this whole pipeline to see where things went awry. The error is then often very easy to solve.

To demonstrate this, we will use the following script that (tries to) fine-tune a DistilBERT model on the [MNLI dataset](https://huggingface.co/datasets/glue):

```py
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
)

raw_datasets = load_dataset("glue", "mnli")

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["premise"], examples["hypothesis"], truncation=True)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

train_dataset = tokenized_datasets["train"].to_tf_dataset(
    columns=["input_ids", "labels"], batch_size=16, shuffle=True
)

validation_dataset = tokenized_datasets["validation_matched"].to_tf_dataset(
    columns=["input_ids", "labels"], batch_size=16, shuffle=True
)

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

model.fit(train_dataset)
```

If you try to execute it, you might get some `VisibleDeprecationWarning`s when doing the dataset conversion -- this is a known UX issue we have, so please ignore it. If you're reading the course after, say, November 2021 and it's still happening, then send rage tweets at @carrigmat until he fixes it.

What's a more serious problem, though, is that we get an outright error. And it's really, terrifyingly long:

```python out
ValueError: No gradients provided for any variable: ['tf_distil_bert_for_sequence_classification/distilbert/embeddings/word_embeddings/weight:0', '...']
```

What does that mean? We tried to train on our data, but we got no gradient? This is pretty perplexing; how do we even begin to debug something like that? When the error you get doesn't immediately suggest where the problem is, the best solution is often to walk through things in sequence, making sure at each stage that everything looks right. And of course, the place to start is always to...

### Check your data

This goes without saying, but if your data is corrupted, Keras is not going to be able to fix it for you. So first things first, you need to have a look at what is inside your training set.

Although it's tempting to look inside `raw_datasets` and `tokenized_datasets`, we highly recommend you go to the data right at the point where it's going to enter the model. That means reading an output from the `tf.data.Dataset` you created with the `to_tf_dataset()` function! So how do we do that? `tf.data.Dataset` objects give us whole batches at a time and don't support indexing, so we can't just ask for `train_dataset[0]`. We can, however, ask it politely for a batch:

```py
for batch in train_dataset:
    break
```

`break` ends the loop after one iteration, so this grabs the first batch that comes out of `train_dataset` and saves it as `batch`. Now, let's take a look at what's inside:

```python out
{'attention_mask': <tf.Tensor: shape=(16, 76), dtype=int64, numpy=
 array([[1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0],
        ...,
        [1, 1, 1, ..., 1, 1, 1],
        [1, 1, 1, ..., 0, 0, 0],
        [1, 1, 1, ..., 0, 0, 0]])>,
 'label': <tf.Tensor: shape=(16,), dtype=int64, numpy=array([0, 2, 1, 2, 1, 1, 2, 0, 0, 0, 1, 0, 1, 2, 2, 1])>,
 'input_ids': <tf.Tensor: shape=(16, 76), dtype=int64, numpy=
 array([[ 101, 2174, 1010, ...,    0,    0,    0],
        [ 101, 3174, 2420, ...,    0,    0,    0],
        [ 101, 2044, 2048, ...,    0,    0,    0],
        ...,
        [ 101, 3398, 3398, ..., 2051, 2894,  102],
        [ 101, 1996, 4124, ...,    0,    0,    0],
        [ 101, 1999, 2070, ...,    0,    0,    0]])>}
```

This looks right, doesn't it? We're passing the `labels`, `attention_mask`, and `input_ids` to the model, which should be everything it needs to compute outputs and calculate the loss. So why don't we have a gradient? Look closer: we're passing a single dictionary as input, but a training batch is usually an input tensor or dictionary, plus a labels tensor. Our labels are just a key in our input dictionary.

Is this a problem? Not always, actually! But it's one of the most common issues you'll encounter when training Transformer models with TensorFlow. Our models can all compute loss internally, but to do that the labels need to be passed in the input dictionary. This is the loss that is used when we don't specify a loss value to `compile()`. Keras, on the other hand, usually expects labels to be passed separately from the input dictionary, and loss computations will usually fail if you don't do that.

The problem has now become clearer: we passed a `loss` argument, which means we're asking Keras to compute losses for us, but we passed our labels as inputs to the model, not as labels in the place Keras expects them! We need to choose one or the other: either we use the model's internal loss and keep the labels where they are, or we keep using Keras losses, but we move the labels to the place Keras expects them. For simplicity, let's take the first approach. Change the call to `compile()` to read:

```py
model.compile(optimizer="adam")
```

Now we'll use the model's internal loss, and this problem should be resolved!

<Tip>

✏️ **Your turn!** As an optional challenge after we've resolved the other issues, you can try coming back to this step and getting the model to work with the original Keras-computed loss instead of the internal loss. You'll need to add `"labels"` to the `label_cols` argument of `to_tf_dataset()` to ensure that the labels are correctly outputted, which will get you gradients -- but there's one more problem with the loss that we specified. Training will still run with this problem, but learning will be very slow and will plateau at a high training loss. Can you figure out what it is?

A ROT13-encoded hint, if you're stuck: Vs lbh ybbx ng gur bhgchgf bs FrdhraprPynffvsvpngvba zbqryf va Genafsbezref, gurve svefg bhgchg vf `ybtvgf`. Jung ner ybtvgf?

And a second hint: Jura lbh fcrpvsl bcgvzvmref, npgvingvbaf be ybffrf jvgu fgevatf, Xrenf frgf nyy gur nethzrag inyhrf gb gurve qrsnhygf. Jung nethzragf qbrf FcnefrPngrtbevpnyPebffragebcl unir, naq jung ner gurve qrsnhygf?

</Tip>

Now, let's try training. We should get gradients now, so hopefully (ominous music plays here) we can just call `model.fit()` and everything will work fine!

```python out
  246/24543 [..............................] - ETA: 15:52 - loss: nan
```

Oh no. 

`nan` is not a very encouraging loss value. Still, we've checked our data, and it looks pretty good. If that's not the problem, where can we go next? The obvious next step is to...

### Check your model

`model.fit()` is a really great convenience function in Keras, but it does a lot of things for you, and that can make it trickier to find exactly where a problem has occurred. If you're debugging your model, one strategy that can really help is to pass just a single batch to the model, and look at the outputs for that one batch in detail. Another really helpful tip if the model is throwing errors is to `compile()` the model with `run_eagerly=True`. This will make it a lot slower, but it will make the error messages much more comprehensible, because they'll indicate exactly where in your model's code the problem occurred.

For now, though, we don't need `run_eagerly` just yet. Let's run the `batch` we got before through the model and see what the outputs look like:

```py
model(batch)
```

```python out
TFSequenceClassifierOutput(loss=<tf.Tensor: shape=(16,), dtype=float32, numpy=
array([nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan, nan,
       nan, nan, nan], dtype=float32)>, logits=<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
array([[nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan],
       [nan, nan]], dtype=float32)>, hidden_states=None, attentions=None)
```

Well, this is tricky. Everything is `nan`! But that's strange, isn't it? How would all our logits become `nan`? `nan` means "not a number." `nan` values often occur when you perform a forbidden operation, such as division by zero. But one thing that's very important to know about `nan` in machine learning is that this value tends to *propagate*. If you multiply a number by `nan`, the output is also `nan`. And if you get a `nan` anywhere in your output, your loss, or your gradient, then it will rapidly spread throughout your whole model -- because when that `nan` value is propagated back through your network, you'll get `nan` gradients, and when weight updates are computed with those gradients, you'll get `nan` weights, and those weights will compute even more `nan` outputs! Soon enough the whole network will just be one big block of `nan`s. Once that happens, it's pretty hard to see where the problem started. How can we isolate where `nan` first crept in?

The answer is to try *reinitializing* our model. Once we started training, we got a `nan` somewhere and it quickly propagated through the whole model. So, let's load the model from a checkpoint and not do any weight updates, and see where we get a `nan` value:

```py
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model(batch)
```

When we run that, we get:

```py out
TFSequenceClassifierOutput(loss=<tf.Tensor: shape=(16,), dtype=float32, numpy=
array([0.6844486 ,        nan,        nan, 0.67127866, 0.7068601 ,
              nan, 0.69309855,        nan, 0.65531296,        nan,
              nan,        nan, 0.675402  ,        nan,        nan,
       0.69831556], dtype=float32)>, logits=<tf.Tensor: shape=(16, 2), dtype=float32, numpy=
array([[-0.04761693, -0.06509043],
       [-0.0481936 , -0.04556257],
       [-0.0040929 , -0.05848458],
       [-0.02417453, -0.0684005 ],
       [-0.02517801, -0.05241832],
       [-0.04514256, -0.0757378 ],
       [-0.02656011, -0.02646275],
       [ 0.00766164, -0.04350497],
       [ 0.02060014, -0.05655622],
       [-0.02615328, -0.0447021 ],
       [-0.05119278, -0.06928903],
       [-0.02859691, -0.04879177],
       [-0.02210129, -0.05791225],
       [-0.02363213, -0.05962167],
       [-0.05352269, -0.0481673 ],
       [-0.08141848, -0.07110836]], dtype=float32)>, hidden_states=None, attentions=None)
```

*Now* we're getting somewhere! There are no `nan` values in our logits, which is reassuring. But we do see a few `nan` values in our loss! Is there something about those samples in particular that's causing this problem? Let's see which ones they are (note that if you run this code yourself, you may get different indices because the dataset has been shuffled):

```python
import numpy as np

loss = model(batch).loss.numpy()
indices = np.flatnonzero(np.isnan(loss))
indices
```

```python out
array([ 1,  2,  5,  7,  9, 10, 11, 13, 14])
```

Let's look at the samples these indices came from:

```python
input_ids = batch["input_ids"].numpy()
input_ids[indices]
```

```python out
array([[  101,  2007,  2032,  2001,  1037, 16480,  3917,  2594,  4135,
        23212,  3070,  2214, 10170,  1010,  2012,  4356,  1997,  3183,
         6838, 12953,  2039,  2000,  1996,  6147,  1997,  2010,  2606,
         1012,   102,  6838,  2001,  3294,  6625,  3773,  1996,  2214,
         2158,  1012,   102,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1998,  6814,  2016,  2234,  2461,  2153,  1998, 13322,
         2009,  1012,   102,  2045,  1005,  1055,  2053,  3382,  2008,
         2016,  1005,  2222,  3046,  8103,  2075,  2009,  2153,  1012,
          102,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1998,  2007,  1996,  3712,  4634,  1010,  2057,  8108,
         2025,  3404,  2028,  1012,  1996,  2616, 18449,  2125,  1999,
         1037,  9666,  1997,  4100,  8663, 11020,  6313,  2791,  1998,
         2431,  1011,  4301,  1012,   102,  2028,  1005,  1055,  5177,
         2110,  1998,  3977,  2000,  2832,  2106,  2025,  2689,  2104,
         2122,  6214,  1012,   102,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1045,  2001,  1999,  1037, 13090,  5948,  2007,  2048,
         2308,  2006,  2026,  5001,  2043,  2026,  2171,  2001,  2170,
         1012,   102,  1045,  2001,  3564,  1999,  2277,  1012,   102,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  2195,  4279,  2191,  2039,  1996,  2181,  2124,  2004,
         1996,  2225,  7363,  1012,   102,  2045,  2003,  2069,  2028,
         2451,  1999,  1996,  2225,  7363,  1012,   102,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  2061,  2008,  1045,  2123,  1005,  1056,  2113,  2065,
         2009,  2428, 10654,  7347,  2030,  2009,  7126,  2256,  2495,
         2291,   102,  2009,  2003,  5094,  2256,  2495,  2291,  2035,
         2105,  1012,   102,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  2051,  1010,  2029,  3216,  2019,  2503,  3444,  1010,
         6732,  1996,  2265,  2038, 19840,  2098,  2125,  9906,  1998,
         2003,  2770,  2041,  1997,  4784,  1012,   102,  2051,  6732,
         1996,  2265,  2003,  9525,  1998,  4569,  1012,   102,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101,  1996, 10556,  2140, 11515,  2058,  1010,  2010,  2162,
         2252,  5689,  2013,  2010,  7223,  1012,   102,  2043,  1996,
        10556,  2140, 11515,  2058,  1010,  2010,  2252,  3062,  2000,
         1996,  2598,  1012,   102,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0],
       [  101, 13543,  1999,  2049,  6143,  2933,  2443,   102,  2025,
        13543,  1999,  6143,  2933,  2003,  2443,   102,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0,     0,     0,     0,     0,     0,
            0,     0,     0,     0]])
```

Well, there's a lot in here, but nothing stands out as unusual. Let's look at the labels:

```python out
labels = batch['labels'].numpy()
labels[indices]
```

```python out
array([2, 2, 2, 2, 2, 2, 2, 2, 2])
```

Ah! The `nan` samples all have the same label, and it's label 2. This is a very strong hint. The fact that we're only getting a loss of `nan` when our label is 2 suggests that this is a very good time to check the number of labels in our model:

```python
model.config.num_labels
```

```python out
2
```

Now we see the problem: the model thinks there are only two classes, but the labels go up to 2, which means there are in fact three classes (because 0 is also a class). This is how we got a `nan` -- by trying to compute the loss for a nonexistent class! Let's try changing that and fitting the model again:

```
model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=3)
model.compile(optimizer='adam')
model.fit(train_dataset)
```

```python out
  869/24543 [>.............................] - ETA: 15:29 - loss: 1.1032
```

We're training! No more `nan`s, and our loss is declining... sort of. If you watch it for a while, you might start to get a bit impatient, because the loss value stays stubbornly high. Let's stop training here and try to think about what could be causing this problem. At this point, we're pretty sure both the data and the model are okay, but our model isn't learning well. What else is left? It's time to...

### Check your hyperparameters

If you look back at the code above, you might not be able to see any hyperparameters at all, except perhaps the `batch_size`, and that doesn't seem like a likely culprit. Don't be fooled, though; there are always hyperparameters, and if you can't see them, it just means that you don't know what they're set to. In particular, remember a critical thing about Keras: if you set a loss, optimizer, or activation function with a string, _all of its arguments will be set to their default values_. This means that even though using strings for this is very convenient, you should be very careful when doing so, as it can easily hide critical things from you. (Anyone trying the optional challenge above should take careful note of this fact.)

In this case, where have we set an argument with a string? We were setting the loss with a string initially, but we're not doing that anymore. We are, however, setting the optimizer with a string. Could that be hiding anything from us? Let's take a look at [its arguments](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam).

Does anything stand out here? That's right -- the learning rate! When we just use the string `'adam'`, we're going to get the default learning rate, which is 0.001, or 1e-3. This is way too high for a Transformer model! In general, we recommend trying learning rates between 1e-5 and 1e-4 for your models; that's somewhere between 10X and 100X smaller than the value we're actually using here. That sounds like it might be a major problem, so let's try reducing it. To do that, we need to import the actual `optimizer` object. While we're at it, let's reinitialize the model from the checkpoint, in case training with the high learning rate damaged its weights:

```python
from tensorflow.keras.optimizers import Adam

model = TFAutoModelForSequenceClassification.from_pretrained(model_checkpoint)
model.compile(optimizer=Adam(5e-5))
```

<Tip>

💡 You can also import the `create_optimizer()` function from 🤗 Transformers, which will give you an AdamW optimizer with correct weight decay as well as learning rate warmup and decay. This optimizer will often produce slightly better results than the ones you get with the default Adam optimizer.

</Tip>

Now, we can try fitting the model with the new, improved learning rate:

```python
model.fit(train_dataset)
```

```python out
319/24543 [..............................] - ETA: 16:07 - loss: 0.9718
```

Now our loss is really going somewhere! Training finally looks like it's working. There's a lesson here: when your model is running but loss isn't declining, and you're sure your data is okay, it's a good idea to check hyperparameters like the learning rate and weight decay. Setting either of those too high is very likely to cause training to "stall" at a high loss value.

## Other potential issues 

We've covered the issues in the script above, but there are several other common errors you might face. Let's take a look at a (very incomplete) list.

### Dealing with out-of-memory errors

The telltale sign of running out of memory is an error like "OOM when allocating tensor" -- OOM is short for "out of memory." This is a very common hazard when dealing with large language models. If you encounter this, a good strategy is to halve your batch size and try again. Bear in mind, though, that some models are *very* large. For example, the full-size GPT-2 has 1.5B parameters, which means you'll need 6 GB of memory just to store the model, and another 6 GB for its gradients! Training the full GPT-2 model will usually require over 20 GB of VRAM no matter what batch size you use, which only a few GPUs have. More lightweight models like `distilbert-base-cased` are much easier to run, and train much more quickly too.

<Tip>

In the next part of the course, we'll look at more advanced techniques that can help you reduce your memory footprint and let you fine-tune the biggest models.

</Tip>

### Hungry Hungry TensorFlow 🦛

One particular quirk of TensorFlow that you should be aware of is that it allocates *all* of your GPU memory to itself as soon as you load a model or do any training, and then it divides up that memory as required. This is different from the behavior of other frameworks, like PyTorch, which allocate memory as required with CUDA rather than doing it internally. One advantage of the TensorFlow approach is that it can often give useful errors when you run out of memory, and it can recover from that state without crashing the whole CUDA kernel. But there's also an important downside: if you run two TensorFlow processes at once, then **you're going to have a bad time**.

If you're running on Colab you don't need to worry about this, but if you're running locally this is definitely something you should be careful about. In particular, be aware that closing a notebook tab does not necessarily shut that notebook down! You may need to select running notebooks (the ones with a green icon) and manually shut them down in the directory listing. Any running notebook that was using TensorFlow could still be holding on to a bunch of your GPU memory, and that means any new notebook you start may encounter some very odd issues.

If you start getting errors about CUDA, BLAS, or cuBLAS in code that worked before, this is very often the culprit. You can use a command like `nvidia-smi` to check -- when you shut down or restart your current notebook, is most of your memory free, or is it still in use? If it's still in use, something else is holding on to it!


### Check your data (again!)

Your model will only learn something if it's actually possible to learn anything from your data. If there is a bug that corrupts the data or the labels are attributed randomly, it's very likely you won't get any model training on your dataset. One helpful tool here is `tokenizer.decode()`. This will turn `input_ids` back into strings, so you can view the data and see if your training data is teaching what you want it to teach. For example, after you get a `batch` from your `tf.data.Dataset` like we did above, you can decode the first element like so:

```py
input_ids = batch["input_ids"].numpy()
tokenizer.decode(input_ids[0])
```

Then you can compare it with the first label, like so:

```py
labels = batch["labels"].numpy()
label = labels[0]
```

Once you can view your data like this, you can ask yourself the following questions:

- Is the decoded data understandable?
- Do you agree with the labels?
- Is there one label that's more common than the others?
- What should the loss/metric be if the model predicted a random answer/always the same answer?

After looking at your data, go through a few of the model's predictions -- if your model outputs tokens, try decoding them too! If the model is always predicting the same thing it might be because your dataset is biased toward one category (for classification problems), so techniques like oversampling rare classes might help. Alternatively, this can also be caused by training issues like bad hyperparameter settings.

If the loss/metric you get on your initial model before any training is very different from the loss/metric you would expect for random predictions, double-check the way your loss or metric is computed, as there is probably a bug there. If you are using several losses that you add at the end, make sure they are of the same scale.

When you are sure your data is perfect, you can see if the model is capable of training on it with one simple test.

### Overfit your model on one batch

Overfitting is usually something we try to avoid when training, as it means the model is not learning to recognize the general features we want it to but is instead just memorizing the training samples. However, trying to train your model on one batch over and over again is a good test to check if the problem as you framed it can be solved by the model you are attempting to train. It will also help you see if your initial learning rate is too high.

Doing this once you have defined your `model` is really easy; just grab a batch of training data, then treat that `batch` as your entire dataset, fitting on it for a large number of epochs:

```py
for batch in train_dataset:
    break

# Make sure you have run model.compile() and set your optimizer,
# and your loss/metrics if you're using them

model.fit(batch, epochs=20)
```

<Tip>

💡 If your training data is unbalanced, make sure to build a batch of training data containing all the labels.

</Tip>

The resulting model should have close-to-perfect results on the `batch`, with a loss declining quickly toward 0 (or the minimum value for the loss you're using).

If you don't manage to have your model obtain perfect results like this, it means there is something wrong with the way you framed the problem or your data, so you should fix that. Only when you manage to pass the overfitting test can you be sure that your model can actually learn something.

<Tip warning={true}>

⚠️ You will have to recreate your model and recompile after this overfitting test, as the model obtained probably won't be able to recover and learn something useful on your full dataset.

</Tip>

### Don't tune anything until you have a first baseline

Intense hyperparameter tuning is always emphasized as being the hardest part of machine learning, but it's just the last step to help you gain a little bit on the metric. *Very* bad values for your hyperparameters, like using the default Adam learning rate of 1e-3 with a Transformer model, will make learning proceed very slowly or completely stall, of course, but most of the time "reasonable" hyperparameters, like a learning rate from 1e-5 to 5e-5, will work just fine to give you good results. So, don't launch into a time-consuming and costly hyperparameter search until you have something that beats the baseline you have on your dataset.

Once you have a good enough model, you can start tweaking a bit. Don't try launching a thousand runs with different hyperparameters, but compare a couple of runs with different values for one hyperparameter to get an idea of which has the greatest impact.

If you are tweaking the model itself, keep it simple and don't try anything you can't reasonably justify. Always make sure you go back to the overfitting test to verify that your change hasn't had any unintended consequences.

### Ask for help

Hopefully you will have found some advice in this section that helped you solve your issue, but if that's not the case, remember you can always ask the community on the [forums](https://discuss.huggingface.co/). 

Here are some additional resources that may prove helpful:

- ["Reproducibility as a vehicle for engineering best practices"](https://docs.google.com/presentation/d/1yHLPvPhUs2KGI5ZWo0sU-PKU3GimAk3iTsI38Z-B5Gw/edit#slide=id.p) by Joel Grus
- ["Checklist for debugging neural networks"](https://towardsdatascience.com/checklist-for-debugging-neural-networks-d8b2a9434f21) by Cecelia Shao
- ["How to unit test machine learning code"](https://medium.com/@keeper6928/how-to-unit-test-machine-learning-code-57cf6fd81765) by Chase Roberts
- ["A Recipe for Training Neural Networks"](http://karpathy.github.io/2019/04/25/recipe/) by Andrej Karpathy

Of course, not every problem you encounter when training neural nets is your own fault! If you encounter something in the 🤗 Transformers or 🤗 Datasets library that does not seem right, you may have encountered a bug. You should definitely tell us all about it, and in the next section we'll explain exactly how to do that.
