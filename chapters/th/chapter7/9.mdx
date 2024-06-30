<FrameworkSwitchCourse {fw} />

<!-- DISABLE-FRONTMATTER-SECTIONS -->

# End-of-chapter quiz[[end-of-chapter-quiz]]

<CourseFloatingBanner
    chapter={7}
    classNames="absolute z-10 right-0 top-0"
/>

Let's test what you learned in this chapter!

### 1. Which of the following tasks can be framed as a token classification problem?

<Question
	choices={[
		{
			text: "Find the grammatical components in a sentence.",
			explain: "Correct! We can then label each word as a noun, verb, etc.",
			correct: true
		},
		{
			text: "Find whether a sentence is grammatically correct or not.",
			explain: "No, this is a sequence classification problem."
		},
		{
			text: "Find the persons mentioned in a sentence.",
			explain: "Correct! We can label each word as person or not person.",
            correct: true
		},
        {
			text: "Find the chunk of words in a sentence that answers a question.",
			explain: "No, that would be a question answering problem."
		}
	]}
/>

### 2. What part of the preprocessing for token classification differs from the other preprocessing pipelines?

<Question
	choices={[
		{
			text: "There is no need to do anything; the texts are already tokenized.",
			explain: "The texts are indeed given as separate words, but we still need to apply the subword tokenization model."
		},
		{
			text: "The texts are given as words, so we only need to apply subword tokenization.",
			explain: "Correct! This is different from the usual preprocessing, where we need to apply the full tokenization pipeline. Can you think of another difference?",
			correct: true
		},
		{
			text: "We use <code>-100</code> to label the special tokens.",
			explain: "That's not specific to token classification -- we always use <code>-100</code> as the label for tokens we want to ignore in the loss."
		},
		{
			text: "We need to make sure to truncate or pad the labels to the same size as the inputs, when applying truncation/padding.",
			explain: "Indeed! That's not the only difference, though.",
			correct: true
		}
	]}
/>

### 3. What problem arises when we tokenize the words in a token classification problem and want to label the tokens?

<Question
	choices={[
		{
			text: "The tokenizer adds special tokens and we have no labels for them.",
			explain: "We label these <code>-100</code> so they are ignored in the loss."
		},
		{
			text: "Each word can produce several tokens, so we end up with more tokens than we have labels.",
			explain: "That is the main problem, and we need to align the original labels with the tokens.",
			correct: true
		},
		{
			text: "The added tokens have no labels, so there is no problem.",
			explain: "That's incorrect; we need as many labels as we have tokens or our models will error out."
		}
	]}
/>

### 4. What does "domain adaptation" mean?

<Question
	choices={[
		{
			text: "It's when we run a model on a dataset and get the predictions for each sample in that dataset.",
			explain: "No, this is just running inference."
		},
		{
			text: "It's when we train a model on a dataset.",
			explain: "No, this is training a model; there is no adaptation here."
		},
		{
			text: "It's when we fine-tune a pretrained model on a new dataset, and it gives predictions that are more adapted to that dataset",
			explain: "Correct! The model adapted its knowledge to the new dataset.",
            correct: true
		},
        {
			text: "It's when we add misclassified samples to a dataset to make our model more robust.",
			explain: "That's certainly something you should do if you retrain your model regularly, but it's not domain adaptation."
		}
	]}
/>

### 5. What are the labels in a masked language modeling problem?

<Question
	choices={[
		{
			text: "Some of the tokens in the input sentence are randomly masked and the labels are the original input tokens.",
			explain: "That's it!",
            correct: true
		},
		{
			text: "Some of the tokens in the input sentence are randomly masked and the labels are the original input tokens, shifted to the left.",
			explain: "No, shifting the labels to the left corresponds to predicting the next word, which is causal language modeling."
		},
		{
			text: "Some of the tokens in the input sentence are randomly masked, and the label is whether the sentence is positive or negative.",
			explain: "That's a sequence classification problem with some data augmentation, not masked language modeling."
		},
        {
			text: "Some of the tokens in the two input sentences are randomly masked, and the label is whether the two sentences are similar or not.",
			explain: "That's a sequence classification problem with some data augmentation, not masked language modeling."
		}
	]}
/>

### 6. Which of these tasks can be seen as a sequence-to-sequence problem?

<Question
	choices={[
		{
			text: "Writing short reviews of long documents",
			explain: "Yes, that's a summarization problem. Try another answer!",
            correct: true
		},
		{
			text: "Answering questions about a document",
			explain: "This can be framed as a sequence-to-sequence problem. It's not the only right answer, though.",
            correct: true
		},
		{
			text: "Translating a text in Chinese into English",
			explain: "That's definitely a sequence-to-sequence problem. Can you spot another one?",
            correct: true
		},
        {
			text: "Fixing the messages sent by my nephew/friend so they're in proper English",
			explain: "That's a kind of translation problem, so definitely a sequence-to-sequence task. This isn't the only right answer, though!",
			correct: true
		}
	]}
/>

### 7. What is the proper way to preprocess the data for a sequence-to-sequence problem?

<Question
	choices={[
		{
			text: "The inputs and targets have to be sent together to the tokenizer with <code>inputs=...</code> and <code>targets=...</code>.",
			explain: "This might be an API we add in the future, but that's not possible right now."
		},
		{
			text: "The inputs and the targets both have to be preprocessed, in two separate calls to the tokenizer.",
			explain: "That is true, but incomplete. There is something you need to do to make sure the tokenizer processes both properly."
		},
		{
			text: "As usual, we just have to tokenize the inputs.",
			explain: "Not in a sequence classification problem; the targets are also texts we need to convert into numbers!"
		},
        {
			text: "The inputs have to be sent to the tokenizer, and the targets too, but under a special context manager.",
			explain: "That's correct, the tokenizer needs to be put into target mode by that context manager.",
			correct: true
		}
	]}
/>

{#if fw === 'pt'}

### 8. Why is there a specific subclass of `Trainer` for sequence-to-sequence problems?

<Question
	choices={[
		{
			text: "Because sequence-to-sequence problems use a custom loss, to ignore the labels set to <code>-100</code>",
			explain: "That's not a custom loss at all, but the way the loss is always computed."
		},
		{
			text: "Because sequence-to-sequence problems require a special evaluation loop",
			explain: "That's correct. Sequence-to-sequence models' predictions are often run using the <code>generate()</code> method.",
			correct: true
		},
		{
			text: "Because the targets are texts in sequence-to-sequence problems",
			explain: "The <code>Trainer</code> doesn't really care about that since they have been preprocessed before."
		},
        {
			text: "Because we use two models in sequence-to-sequence problems",
			explain: "We do use two models in a way, an encoder and a decoder, but they are grouped together in one model."
		}
	]}
/>

{:else}

### 9. Why is it often unnecessary to specify a loss when calling `compile()` on a Transformer model?

<Question
	choices={[
		{
			text: "Because Transformer models are trained with unsupervised learning",
			explain: "Not quite -- even unsupervised learning needs a loss function!"
		},
		{
			text: "Because the model's internal loss output is used by default",
			explain: "That's correct!",
			correct: true
		},
		{
			text: "Because we compute metrics after training instead",
			explain: "We do often do that, but it doesn't explain where we get the loss value we optimize in training."
		},
        {
			text: "Because loss is specified in `model.fit()` instead",
			explain: "No, the loss function is always fixed once you run `model.compile()`, and can't be changed in `model.fit()`."
		}
	]}
/>

{/if}

### 10. When should you pretrain a new model?

<Question
	choices={[
		{
			text: "When there is no pretrained model available for your specific language",
			explain: "That's correct.",
			correct: true
		},
		{
			text: "When you have lots of data available, even if there is a pretrained model that could work on it",
			explain: "In this case, you should probably use the pretrained model and fine-tune it on your data, to avoid huge compute costs."
		},
		{
			text: "When you have concerns about the bias of the pretrained model you are using",
			explain: "That is true, but you have to make very sure the data you will use for training is really better.",
			correct: true
		},
        {
			text: "When the pretrained models available are just not good enough",
			explain: "Are you sure you've properly debugged your training, then?"
		}
	]}
/>

### 11. Why is it easy to pretrain a language model on lots and lots of texts?

<Question
	choices={[
		{
			text: "Because there are plenty of texts available on the internet",
			explain: "Although true, that doesn't really answer the question. Try again!"
		},
		{
			text: "Because the pretraining objective does not require humans to label the data",
			explain: "That's correct, language modeling is a self-supervised problem.",
			correct: true
		},
		{
			text: "Because the 🤗 Transformers library only requires a few lines of code to start the training",
			explain: "Although true, that doesn't really answer the question asked. Try another answer!"
		}
	]}
/>

### 12. What are the main challenges when preprocessing data for a question answering task?

<Question
	choices={[
		{
			text: "You need to tokenize the inputs.",
			explain: "That's correct, but is it really a main challenge?"
		},
		{
			text: "You need to deal with very long contexts, which give several training features that may or may not have the answer in them.",
			explain: "This is definitely one of the challenges.",
			correct: true
		},
		{
			text: "You need to tokenize the answers to the question as well as the inputs.",
			explain: "No, unless you are framing your question answering problem as a sequence-to-sequence task."
		},
       {
			text: "From the answer span in the text, you have to find the start and end token in the tokenized input.",
			explain: "That's one of the hard parts, yes!",
			correct: true
		}
	]}
/>

### 13. How is post-processing usually done in question answering?

<Question
	choices={[
		{
			text: "The model gives you the start and end positions of the answer, and you just have to decode the corresponding span of tokens.",
			explain: "That could be one way to do it, but it's a bit too simplistic."
		},
		{
			text: "The model gives you the start and end positions of the answer for each feature created by one example, and you just have to decode the corresponding span of tokens in the one that has the best score.",
			explain: "That's close to the post-processing we studied, but it's not entirely right."
		},
		{
			text: "The model gives you the start and end positions of the answer for each feature created by one example, and you just have to match them to the span in the context for the one that has the best score.",
			explain: "That's it in a nutshell!",
			correct: true
		},
        {
			text: "The model generates an answer, and you just have to decode it.",
			explain: "No, unless you are framing your question answering problem as a sequence-to-sequence task."
		}
	]}
/>
