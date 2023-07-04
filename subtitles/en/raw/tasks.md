Note: the following transcripts are associated with Merve Noyan's videos in the Hugging Face Tasks playlist: https://www.youtube.com/playlist?list=PLo2EIpI_JMQtyEr-sLJSy5_SnLCb4vtQf

Token Classification video

Welcome to the Hugging Face tasks series! In this video we’ll take a look at the token classification task.
Token classification is the task of assigning a label to each token in a sentence. There are various token classification tasks and the most common are Named Entity Recognition and Part-of-Speech Tagging.
Let’s take a quick look at the Named Entity Recognition task. The goal of this task is to find the entities in a piece of text, such as person, location, or organization. This task is formulated as labelling each token with one class for each entity, and another class for tokens that have no entity.
Another token classification task is part-of-speech tagging. The goal of this task is to label the words for a particular part of a speech, such as noun, pronoun, adjective, verb and so on. This task is formulated as labelling each token with parts of speech.
Token classification models are evaluated on Accuracy, Recall, Precision and F1-Score. The metrics are calculated for each of the classes. We calculate true positive, true negative and false positives to calculate precision and recall, and take their harmonic mean to get F1-Score. Then we calculate it for every class and take the overall average to evaluate our model.
An example dataset used for this task is ConLL2003. Here, each token belongs to a certain named entity class, denoted as the indices of the list containing the labels.
You can extract important information from invoices using named entity recognition models, such as date, organization name or address.
For more information about the Token classification task, check out the Hugging Face course.


Question Answering video

Welcome to the Hugging Face tasks series. In this video, we will take a look at the Question Answering task.
Question answering is the task of extracting an answer in a given document.
Question answering models take a context, which is the document you want to search in, and a question and return an answer. Note that the answer is not generated, but extracted from the context. This type of question answering is called extractive.
The task is evaluated on two metrics, exact match and F1-Score.
As the name implies, exact match looks for an exact match between the predicted answer and the correct answer.
A common metric used is the F1-Score, which is calculated over tokens that are predicted correctly and incorrectly. It is calculated over the average of two metrics called precision and recall which are metrics that are used widely in classification problems.
An example dataset used for this task is called SQuAD. This dataset contains contexts, questions and the answers that are obtained from English Wikipedia articles.
You can use question answering models to automatically answer the questions asked by your customers. You simply need a document containing information about your business and query through that document with the questions asked by your customers.
For more information about the Question Answering task, check out the Hugging Face course.


Causal Language Modeling video

Welcome to the Hugging Face tasks series! In this video we’ll take a look at Causal Language Modeling.
Causal language modeling is the task of predicting the next 
word in a sentence, given all the previous words. This task is very similar to the autocorrect function that you might have on your phone. 
These models take a sequence to be completed and outputs the complete sequence.
Classification metrics can’t be used as there’s no single correct answer for completion. Instead, we evaluate the distribution of the text completed by the model.
A common metric to do so is the cross-entropy loss. Perplexity is also a widely used metric and it is calculated as the exponential of the cross-entropy loss.
You can use any dataset with plain text and tokenize the text to prepare the data. 
Causal language models can be used to generate code.
For more information about the Causal Language Modeling task, check out the Hugging Face course.


Masked Language Modeling video

Welcome to the Hugging Face tasks series! In this video we’ll take a look at Masked Language Modeling.
Masked language modeling is the task of predicting which words should fill in the blanks of a sentence.
These models take a masked text as the input and output the possible values for that mask.
Masked language modeling is handy before fine-tuning your model for your task. For example, if you need to use a model in a specific domain, say, biomedical documents, models like BERT will treat your domain-specific words as rare tokens. If you train a masked language model using your biomedical corpus and then fine tune your model on a downstream task, you will have a better performance.
Classification metrics can’t be used as there’s no single correct answer to mask values. Instead, we evaluate the distribution of the mask values.
A common metric to do so is the cross-entropy loss. Perplexity is also a widely used metric and it is calculated as the exponential of the cross-entropy loss.
You can use any dataset with plain text and tokenize the text to mask the data.
For more information about the Masked Language Modeling, check out the Hugging Face course.


Summarization video

Welcome to the Hugging Face tasks series. In this video, we will take a look at the Text Summarization task.
Summarization is a task of producing a shorter version of a document while preserving the relevant and important information in the document.
Summarization models take a document to be summarized and output the summarized text.
This task is evaluated on the ROUGE score. It’s based on the overlap between the produced sequence and the correct sequence.
You might see this as ROUGE-1, which is the overlap of single tokens and ROUGE-2, the overlap of subsequent token pairs. ROUGE-N refers to the overlap of n subsequent tokens. Here we see an example of how overlaps take place.
An example dataset used for this task is called Extreme Summarization, XSUM. This dataset contains texts and their summarized versions.
You can use summarization models to summarize research papers which would enable researchers to easily pick papers for their reading list.
For more information about the Summarization task, check out the Hugging Face course.


Translation video

Welcome to the Hugging Face tasks series. In this video, we will take a look at the Translation task.
Translation is the task of translating text from one language to another.
These models take a text in the source language and output the translation of that text in the target language.
The task is evaluated on the BLEU score.
The score ranges from 0 to 1, in which 1 means the translation perfectly matched and 0 did not match at all.
BLEU is calculated over subsequent tokens called n-grams. Unigram refers to a single token while bi-gram refers to token pairs and n-grams refer to n subsequent tokens. 
Machine translation datasets contain pairs of text in a language and translation of the text in another language.
These models can help you build conversational agents across different languages.
One option is to translate the training data used for the chatbot and train a separate chatbot.
You can put one translation model from your user’s language to the language your chatbot is trained on, translate the user inputs and do intent classification, take the output of the chatbot and translate it from the language your chatbot was trained on to the user’s language.
For more information about the Translation task, check out the Hugging Face course.
