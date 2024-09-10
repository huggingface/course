<FrameworkSwitchCourse {fw} />

# Semantic search with FAISS[[semantic-search-with-faiss]]

{#if fw === 'pt'}

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_pt.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_pt.ipynb"},
]} />

{:else}

<CourseFloatingBanner chapter={5}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_tf.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter5/section6_tf.ipynb"},
]} />

{/if}

In [section 5](/course/chapter5/5), we created a dataset of GitHub issues and comments from the 🤗 Datasets repository. In this section we'll use this information to build a search engine that can help us find answers to our most pressing questions about the library!

<Youtube id="OATCgQtNX2o"/>

## Using embeddings for semantic search[[using-embeddings-for-semantic-search]]

As we saw in [Chapter 1](/course/chapter1), Transformer-based language models represent each token in a span of text as an _embedding vector_. It turns out that one can "pool" the individual embeddings to create a vector representation for whole sentences, paragraphs, or (in some cases) documents. These embeddings can then be used to find similar documents in the corpus by computing the dot-product similarity (or some other similarity metric) between each embedding and returning the documents with the greatest overlap.

In this section we'll use embeddings to develop a semantic search engine. These search engines offer several advantages over conventional approaches that are based on matching keywords in a query with the documents.

<div class="flex justify-center">
<img class="block dark:hidden" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/semantic-search.svg" alt="Semantic search."/>
<img class="hidden dark:block" src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter5/semantic-search-dark.svg" alt="Semantic search."/>
</div>

## Loading and preparing the dataset[[loading-and-preparing-the-dataset]]

The first thing we need to do is download our dataset of GitHub issues, so let's use `load_dataset()` function as usual:

```py
from datasets import load_dataset

issues_dataset = load_dataset("lewtun/github-issues", split="train")
issues_dataset
```

```python out
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
    num_rows: 2855
})
```

Here we've specified the default `train` split in `load_dataset()`, so it returns a `Dataset` instead of a `DatasetDict`. The first order of business is to filter out the pull requests, as these tend to be rarely used for answering user queries and will introduce noise in our search engine. As should be familiar by now, we can use the `Dataset.filter()` function to exclude these rows in our dataset. While we're at it, let's also filter out rows with no comments, since these provide no answers to user queries:

```py
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
issues_dataset
```

```python out
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignee', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'active_lock_reason', 'pull_request', 'body', 'performed_via_github_app', 'is_pull_request'],
    num_rows: 771
})
```

We can see that there are a lot of columns in our dataset, most of which we don't need to build our search engine. From a search perspective, the most informative columns are `title`, `body`, and `comments`, while `html_url` provides us with a link back to the source issue. Let's use the `Dataset.remove_columns()` function to drop the rest:

```py
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
issues_dataset
```

```python out
Dataset({
    features: ['html_url', 'title', 'comments', 'body'],
    num_rows: 771
})
```

To create our embeddings we'll augment each comment with the issue's title and body, since these fields often include useful contextual information. Because our `comments` column is currently a list of comments for each issue, we need to "explode" the column so that each row consists of an `(html_url, title, body, comment)` tuple. In Pandas we can do this with the [`DataFrame.explode()` function](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.explode.html), which creates a new row for each element in a list-like column, while replicating all the other column values. To see this in action, let's first switch to the Pandas  `DataFrame` format:

```py
issues_dataset.set_format("pandas")
df = issues_dataset[:]
```

If we inspect the first row in this `DataFrame` we can see there are four comments associated with this issue:

```py
df["comments"][0].tolist()
```

```python out
['the bug code locate in ：\r\n    if data_args.task_name is not None:\r\n        # Downloading and loading a dataset from the hub.\r\n        datasets = load_dataset("glue", data_args.task_name, cache_dir=model_args.cache_dir)',
 'Hi @jinec,\r\n\r\nFrom time to time we get this kind of `ConnectionError` coming from the github.com website: https://raw.githubusercontent.com\r\n\r\nNormally, it should work if you wait a little and then retry.\r\n\r\nCould you please confirm if the problem persists?',
 'cannot connect，even by Web browser，please check that  there is some  problems。',
 'I can access https://raw.githubusercontent.com/huggingface/datasets/1.7.0/datasets/glue/glue.py without problem...']
```

When we explode `df`, we expect to get one row for each of these comments. Let's check if that's the case:

```py
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)
```

<table border="1" class="dataframe" style="table-layout: fixed; word-wrap:break-word; width: 100%;">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>html_url</th>
      <th>title</th>
      <th>comments</th>
      <th>body</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>the bug code locate in ：\r\n    if data_args.task_name is not None...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>Hi @jinec,\r\n\r\nFrom time to time we get this kind of `ConnectionError` coming from the github.com website: https://raw.githubusercontent.com...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>cannot connect，even by Web browser，please check that  there is some  problems。</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>https://github.com/huggingface/datasets/issues/2787</td>
      <td>ConnectionError: Couldn't reach https://raw.githubusercontent.com</td>
      <td>I can access https://raw.githubusercontent.com/huggingface/datasets/1.7.0/datasets/glue/glue.py without problem...</td>
      <td>Hello,\r\nI am trying to run run_glue.py and it gives me this error...</td>
    </tr>
  </tbody>
</table>

Great, we can see the rows have been replicated, with the `comments` column containing the individual comments! Now that we're finished with Pandas, we can quickly switch back to a `Dataset` by loading the `DataFrame` in memory:

```py
from datasets import Dataset

comments_dataset = Dataset.from_pandas(comments_df)
comments_dataset
```

```python out
Dataset({
    features: ['html_url', 'title', 'comments', 'body'],
    num_rows: 2842
})
```

Okay, this has given us a few thousand comments to work with!


<Tip>

✏️ **Try it out!** See if you can use `Dataset.map()` to explode the `comments` column of `issues_dataset` _without_ resorting to the use of Pandas. This is a little tricky; you might find the ["Batch mapping"](https://huggingface.co/docs/datasets/about_map_batch#batch-mapping) section of the 🤗 Datasets documentation useful for this task.

</Tip>

Now that we have one comment per row, let's create a new `comments_length` column that contains the number of words per comment:

```py
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
```

We can use this new column to filter out short comments, which typically include things like "cc @lewtun" or "Thanks!" that are not relevant for our search engine. There's no precise number to select for the filter, but around 15 words seems like a good start:

```py
comments_dataset = comments_dataset.filter(lambda x: x["comment_length"] > 15)
comments_dataset
```

```python out
Dataset({
    features: ['html_url', 'title', 'comments', 'body', 'comment_length'],
    num_rows: 2098
})
```

Having cleaned up our dataset a bit, let's concatenate the issue title, description, and comments together in a new `text` column. As usual, we'll write a simple function that we can pass to `Dataset.map()`:

```py
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }


comments_dataset = comments_dataset.map(concatenate_text)
```

We're finally ready to create some embeddings! Let's take a look.

## Creating text embeddings[[creating-text-embeddings]]

We saw in [Chapter 2](/course/chapter2) that we can obtain token embeddings by using the `AutoModel` class. All we need to do is pick a suitable checkpoint to load the model from. Fortunately, there's a library called `sentence-transformers` that is dedicated to creating embeddings. As described in the library's [documentation](https://www.sbert.net/examples/applications/semantic-search/README.html#symmetric-vs-asymmetric-semantic-search), our use case is an example of _asymmetric semantic search_ because we have a short query whose answer we'd like to find in a longer document, like a an issue comment. The handy [model overview table](https://www.sbert.net/docs/pretrained_models.html#model-overview) in the documentation indicates that the `multi-qa-mpnet-base-dot-v1` checkpoint has the best performance for semantic search, so we'll use that for our application. We'll also load the tokenizer using the same checkpoint:

{#if fw === 'pt'}

```py
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)
```

To speed up the embedding process, it helps to place the model and inputs on a GPU device, so let's do that now:

```py
import torch

device = torch.device("cuda")
model.to(device)
```

{:else}

```py
from transformers import AutoTokenizer, TFAutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = TFAutoModel.from_pretrained(model_ckpt, from_pt=True)
```

Note that we've set `from_pt=True` as an argument of the `from_pretrained()` method. That's because the `multi-qa-mpnet-base-dot-v1` checkpoint only has PyTorch weights, so setting `from_pt=True` will automatically convert them to the TensorFlow format for us. As you can see, it is very simple to switch between frameworks in 🤗 Transformers!

{/if}

As we mentioned earlier, we'd like to represent each entry in our GitHub issues corpus as a single vector, so we need to "pool" or average our token embeddings in some way. One popular approach is to perform *CLS pooling* on our model's outputs, where we simply collect the last hidden state for the special `[CLS]` token. The following function does the trick for us:

```py
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]
```

Next, we'll create a helper function that will tokenize a list of documents, place the tensors on the GPU, feed them to the model, and finally apply CLS pooling to the outputs:

{#if fw === 'pt'}

```py
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
```

We can test the function works by feeding it the first text entry in our corpus and inspecting the output shape:

```py
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
```

```python out
torch.Size([1, 768])
```

Great, we've converted the first entry in our corpus into a 768-dimensional vector! We can use `Dataset.map()` to apply our `get_embeddings()` function to each row in our corpus, so let's create a new `embeddings` column as follows:

```py
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
```

{:else}

```py
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="tf"
    )
    encoded_input = {k: v for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)
```

We can test the function works by feeding it the first text entry in our corpus and inspecting the output shape:

```py
embedding = get_embeddings(comments_dataset["text"][0])
embedding.shape
```

```python out
TensorShape([1, 768])
```

Great, we've converted the first entry in our corpus into a 768-dimensional vector! We can use `Dataset.map()` to apply our `get_embeddings()` function to each row in our corpus, so let's create a new `embeddings` column as follows:

```py
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).numpy()[0]}
)
```

{/if}

Notice that we've converted the embeddings to NumPy arrays -- that's because 🤗 Datasets requires this format when we try to index them with FAISS, which we'll do next.


## Using FAISS for efficient similarity search[[using-faiss-for-efficient-similarity-search]]

Now that we have a dataset of embeddings, we need some way to search over them. To do this, we'll use a special data structure in 🤗 Datasets called a _FAISS index_. [FAISS](https://faiss.ai/) (short for Facebook AI Similarity Search) is a library that provides efficient algorithms to quickly search and cluster embedding vectors.

The basic idea behind FAISS is to create a special data structure called an _index_ that allows one to find which embeddings are similar to an input embedding. Creating a FAISS index in 🤗 Datasets is simple -- we use the `Dataset.add_faiss_index()` function and specify which column of our dataset we'd like to index:

```py
embeddings_dataset.add_faiss_index(column="embeddings")
```

We can now perform queries on this index by doing a nearest neighbor lookup with the `Dataset.get_nearest_examples()` function. Let's test this out by first embedding a question as follows:

{#if fw === 'pt'}

```py
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()
question_embedding.shape
```

```python out
torch.Size([1, 768])
```

{:else}

```py
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).numpy()
question_embedding.shape
```

```python out
(1, 768)
```

{/if}

Just like with the documents, we now have a 768-dimensional vector representing the query, which we can compare against the whole corpus to find the most similar embeddings:

```py
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)
```

The `Dataset.get_nearest_examples()` function returns a tuple of scores that rank the overlap between the query and the document, and a corresponding set of samples (here, the 5 best matches). Let's collect these in a `pandas.DataFrame` so we can easily sort them:

```py
import pandas as pd

samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)
```

Now we can iterate over the first few rows to see how well our query matched the available comments:

```py
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()
```

```python out
"""
COMMENT: Requiring online connection is a deal breaker in some cases unfortunately so it'd be great if offline mode is added similar to how `transformers` loads models offline fine.

@mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?
SCORE: 25.505046844482422
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: The local dataset builders (csv, text , json and pandas) are now part of the `datasets` package since #1726 :)
You can now use them offline
\`\`\`python
datasets = load_dataset("text", data_files=data_files)
\`\`\`

We'll do a new release soon
SCORE: 24.555509567260742
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: I opened a PR that allows to reload modules that have already been loaded once even if there's no internet.

Let me know if you know other ways that can make the offline mode experience better. I'd be happy to add them :)

I already note the "freeze" modules option, to prevent local modules updates. It would be a cool feature.

----------

> @mandubian's second bullet point suggests that there's a workaround allowing you to use your offline (custom?) dataset with `datasets`. Could you please elaborate on how that should look like?

Indeed `load_dataset` allows to load remote dataset script (squad, glue, etc.) but also you own local ones.
For example if you have a dataset script at `./my_dataset/my_dataset.py` then you can do
\`\`\`python
load_dataset("./my_dataset")
\`\`\`
and the dataset script will generate your dataset once and for all.

----------

About I'm looking into having `csv`, `json`, `text`, `pandas` dataset builders already included in the `datasets` package, so that they are available offline by default, as opposed to the other datasets that require the script to be downloaded.
cf #1724
SCORE: 24.14896583557129
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: > here is my way to load a dataset offline, but it **requires** an online machine
>
> 1. (online machine)
>
> ```
>
> import datasets
>
> data = datasets.load_dataset(...)
>
> data.save_to_disk(/YOUR/DATASET/DIR)
>
> ```
>
> 2. copy the dir from online to the offline machine
>
> 3. (offline machine)
>
> ```
>
> import datasets
>
> data = datasets.load_from_disk(/SAVED/DATA/DIR)
>
> ```
>
>
>
> HTH.


SCORE: 22.893993377685547
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================

COMMENT: here is my way to load a dataset offline, but it **requires** an online machine
1. (online machine)
\`\`\`
import datasets
data = datasets.load_dataset(...)
data.save_to_disk(/YOUR/DATASET/DIR)
\`\`\`
2. copy the dir from online to the offline machine
3. (offline machine)
\`\`\`
import datasets
data = datasets.load_from_disk(/SAVED/DATA/DIR)
\`\`\`

HTH.
SCORE: 22.406635284423828
TITLE: Discussion using datasets in offline mode
URL: https://github.com/huggingface/datasets/issues/824
==================================================
"""
```

Not bad! Our second hit seems to match the query.

<Tip>

✏️ **Try it out!** Create your own query and see whether you can find an answer in the retrieved documents. You might have to increase the `k` parameter in `Dataset.get_nearest_examples()` to broaden the search.

</Tip>