<!-- DISABLE-FRONTMATTER-SECTIONS -->

# End-of-chapter quiz[[end-of-chapter-quiz]]

<CourseFloatingBanner
    chapter={5}
    classNames="absolute z-10 right-0 top-0"
/>

This chapter covered a lot of ground! Don't worry if you didn't grasp all the details; the next chapters will help you understand how things work under the hood.

Before moving on, though, let's test what you learned in this chapter.

### 1. The `load_dataset()` function in 🤗 Datasets allows you to load a dataset from which of the following locations? 

<Question
	choices={[
		{
			text: "Locally, e.g. on your laptop",
			explain: "Correct! You can pass the paths of local files to the <code>data_files</code> argument of <code>load_dataset()</code> to load local datasets.",
			correct: true
		},
		{
			text: "The Hugging Face Hub",
			explain: "Correct! You can load datasets on the Hub by providing the dataset ID, e.g. <code>load_dataset('emotion')</code>.",
			correct: true
		},
		{
			text: "A remote server",
			explain: "Correct! You can pass URLs to the <code>data_files</code> argument of <code>load_dataset()</code> to load remote files.",
			correct: true
		},
	]}
/>

### 2. Suppose you load one of the GLUE tasks as follows:

```py
from datasets import load_dataset

dataset = load_dataset("glue", "mrpc", split="train")
```

Which of the following commands will produce a random sample of 50 elements from `dataset`?

<Question
	choices={[
		{
			text: "<code>dataset.sample(50)</code>",
			explain: "This is incorrect -- there is no <code>Dataset.sample()</code> method."
		},
		{
			text: "<code>dataset.shuffle().select(range(50))</code>",
			explain: "Correct! As you saw in this chapter, you first shuffle the dataset and then select the samples from it.",
			correct: true
		},
		{
			text: "<code>dataset.select(range(50)).shuffle()</code>",
			explain: "This is incorrect -- although the code will run, it will only shuffle the first 50 elements in the dataset."
		}
	]}
/>

### 3. Suppose you have a dataset about household pets called `pets_dataset`, which has a `name` column that denotes the name of each pet. Which of the following approaches would allow you to filter the dataset for all pets whose names start with the letter "L"?

<Question
	choices={[
		{
			text: "<code>pets_dataset.filter(lambda x : x['name'].startswith('L'))</code>",
			explain: "Correct! Using a Python lambda function for these quick filters is a great idea. Can you think of another solution?",
			correct: true
		},
		{
			text: "<code>pets_dataset.filter(lambda x['name'].startswith('L'))</code>",
			explain: "This is incorrect -- a lambda function takes the general form <code>lambda *arguments* : *expression*</code>, so you need to provide arguments in this case."
		},
		{
			text: "Create a function like <code>def filter_names(x): return x['name'].startswith('L')</code> and run <code>pets_dataset.filter(filter_names)</code>.",
			explain: "Correct! Just like with <code>Dataset.map()</code>, you can pass explicit functions to <code>Dataset.filter()</code>. This is useful when you have some complex logic that isn't suitable for a short lambda function. Which of the other solutions would work?",
			correct: true
		}
	]}
/>

### 4. What is memory mapping?

<Question
	choices={[
		{
			text: "A mapping between CPU and GPU RAM",
			explain: "That's not it -- try again!",
		},
		{
			text: "A mapping between RAM and filesystem storage",
			explain: "Correct! 🤗 Datasets treats each dataset as a memory-mapped file. This allows the library to access and operate on elements of the dataset without needing to fully load it into memory.",
			correct: true
		},
		{
			text: "A mapping between two files in the 🤗 Datasets cache",
			explain: "This is not correct - try again!"
		}
	]}
/>

### 5. Which of the following are the main benefits of memory mapping?

<Question
	choices={[
		{
			text: "Accessing memory-mapped files is faster than reading from or writing to disk.",
			explain: "Correct! This allows 🤗 Datasets to be blazing fast. That's not the only benefit, though.",
			correct: true
		},
		{
			text: "Applications can access segments of data in an extremely large file without having to read the whole file into RAM first.",
			explain: "Correct! This allows 🤗 Datasets to load multi-gigabyte datasets on your laptop without blowing up your CPU. What other advantage does memory mapping offer?",
			correct: true
		},
		{
			text: "It consumes less energy, so your battery lasts longer.",
			explain: "This is not correct -- try again!"
		}
	]}
/>

### 6. Why does the following code fail?

```py
from datasets import load_dataset

dataset = load_dataset("allocine", streaming=True, split="train")
dataset[0]
```

<Question
	choices={[
		{
			text: "It tries to stream a dataset that's too large to fit in RAM.",
			explain: "This is not correct -- streaming datasets are decompressed on the fly, and you can process terabyte-sized datasets with very little RAM!",
		},
		{
			text: "It tries to access an <code>IterableDataset</code>.",
			explain: "Correct! An <code>IterableDataset</code> is a generator, not a container, so you should access its elements using <code>next(iter(dataset))</code>.",
			correct: true
		},
		{
			text: "The <code>allocine</code> dataset doesn't have a <code>train</code> split.",
			explain: "This is incorrect -- check out the [<code>allocine</code> dataset card](https://huggingface.co/datasets/allocine) on the Hub to see which splits it contains."
		}
	]}
/>

### 7. Which of the following are the main benefits of creating a dataset card?

<Question
	choices={[
		{
			text: "It provides information about the intended use and supported tasks of the dataset so others in the community can make an informed decision about using it.",
			explain: "Correct! Undocumented datasets may be used to train models that may not reflect the intentions of the dataset creators, or may produce models whose legal status is murky if they're trained on data that violates privacy or licensing restrictions. This isn't the only benefit, though!",
			correct : true
		},
		{
			text: "It helps draw attention to the biases that are present in a corpus.",
			explain: "Correct! Almost all datasets have some form of bias, which can produce negative consequences downstream. Being aware of them helps model builders understand how to address the inherent biases. What else do dataset cards help with?",
			correct : true
		},
		{
			text: "It improves the chances that others in the community will use my dataset.",
			explain: "Correct! A well-written dataset card will tend to lead to higher usage of your precious dataset. What other benefits does it offer?",
			correct: true
		},
	]}
/>


### 8. What is semantic search?

<Question
	choices={[
		{
			text: "A way to search for exact matches between the words in a query and the documents in a corpus",
			explain: "This is incorrect -- this type of search is called *lexical search*, and it's what you typically see with traditional search engines."
		},
		{
			text: "A way to search for matching documents by understanding the contextual meaning of a query",
			explain: "Correct! Semantic search uses embedding vectors to represent queries and documents, and uses a similarity metric to measure the amount of overlap between them. How else might you describe it?",
			correct: true
		},
		{
			text: "A way to improve search accuracy",
			explain: "Correct! Semantic search engines can capture the intent of a query much better than keyword matching and typically retrieve documents with higher precision. But this isn't the only right answer - what else does semantic search provide?",
			correct: true
		}
	]}
/>

### 9. For asymmetric semantic search, you usually have:

<Question
	choices={[
		{
			text: "A short query and a longer paragraph that answers the query",
			explain: "Correct!",
			correct : true
		},
		{
			text: "Queries and paragraphs that are of about the same length",
			explain: "This is actually an example of symmetric semantic search -- try again!"
		},
		{
			text: "A long query and a shorter paragraph that answers the query",
			explain: "This is incorrect -- try again!"
		}
	]}
/>

### 10. Can I use 🤗 Datasets to load data for use in other domains, like speech processing?

<Question
	choices={[
		{
			text: "No",
			explain: "This is incorrect -- 🤗 Datasets currently supports tabular data, audio, and computer vision. Check out the <a  href='https://huggingface.co/datasets/mnist'>MNIST dataset</a> on the Hub for a computer vision example."
		},
		{
			text: "Yes",
			explain: "Correct! Check out the exciting developments with speech and vision in the 🤗 Transformers library to see how 🤗 Datasets is used in these domains.",
			correct : true
		},
	]}
/>
