Loading a custom dataset. Although the Hugging Face Hub hosts over a thousand public datasets, you'll often need to work with data that is stored on your laptop or some remote server. In this video we'll explore how the Datasets library can be used to load datasets that aren’t available on the Hugging Face Hub.

As you can see in this table, the Datasets library provides several in-built scripts to load datasets in several formats. To load a dataset in one of these formats, you just need to provide the name of the format to the load_dataset function, along with a data_files argument that points to one or more filepaths or URLs.

To see this in action, let's start by loading a local CSV file. In this example, we first download a dataset about wine quality from the UCI machine learning respository.

Since this is a CSV file, we then specify the csv loading script.

This script needs to know where our data is located, so we provide the filename as part of the data_files argument.

The CSV loading script also allows you to pass several keyword arguments, so here we've also specified the separator as a semi-colon. And with that we can see the dataset is loaded automatically as a DatasetDict object, with each column in the CSV file represented as a feature.

If your dataset is located on some remote server like GitHub or some other repository, the process is very similar. The only difference is that now the data_files argument points to a URL instead of a local filepath.

Let's now take a look at loading raw text files. This format is quite common in NLP and you'll typically find books and plays are just a single file with raw text inside. In this example, we have a text file of Shakespere plays that's stored on a GitHub repository. As we did for CSV files, we simply choose the text loading script and point the data_files argument to the URL. As you can see, these files are processed line-by-line, so empty lines in the raw text are also represented as a row in the dataset.

For JSON files, there are two main formats to know about. The first one is called JSON Lines, where every row in the file is a separate JSON object. For these files, you can load the dataset by selecting the json loading script and pointing the data_files argument to the file or URL. In this example, we've loaded a JSON lines files based on Stack Exchange questions and answers.