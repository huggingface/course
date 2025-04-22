class: impact

# Presentation based on 3.mdx
## Generated Presentation

.center[![Hugging Face Logo](https://huggingface.co/front/assets/huggingface_logo.svg)]

???
Welcome everyone. This presentation, automatically generated from the course material titled '3.mdx', will walk you through the key topics discussed in the document. Let's begin.

---

# Transformers, what can they do?

- Introduction to Transformer models and their capabilities.
- Using the `pipeline()` function from the 🤗 Transformers library.

???
In this section, we'll explore the capabilities of Transformer models and introduce the `pipeline()` function from the Hugging Face Transformers library, which simplifies working with these models.

---

# Transformers are everywhere

- Transformer models are used across various domains: NLP, computer vision, audio processing.
- Companies and organizations using Hugging Face and Transformer models.

.center[![Companies using Hugging Face](https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter1/companies.PNG)]

???
Transformer models are ubiquitous, powering applications in natural language processing, computer vision, and audio processing. Many companies and organizations leverage Hugging Face's tools and models, contributing back to the community.

---

# Working with pipelines

- The `pipeline()` function simplifies using Transformer models.
- Example: Sentiment analysis using a pretrained model.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier("I've been waiting for a HuggingFace course my whole life.")
```

Output:
```python
[{'label': 'POSITIVE', 'score': 0.9598047137260437}]
```

???
The `pipeline()` function is a powerful tool in the Hugging Face Transformers library, abstracting away the complexity of working with models. Here, we demonstrate sentiment analysis using a pretrained model.

---

# Available pipelines for different modalities

- Text pipelines: sentiment-analysis, text-generation, etc.
- Image pipelines: image-classification, object-detection, etc.
- Audio pipelines: automatic-speech-recognition, audio-classification, etc.
- Multimodal pipelines: document-question-answering, visual-question-answering.

???
The Hugging Face Transformers library offers a wide range of pipelines for various modalities, including text, images, audio, and multimodal tasks. These pipelines simplify common tasks like sentiment analysis, image classification, and speech recognition.

---

# Zero-shot classification

- Classifying text without prior training on specific labels.
- Example: Classifying text into custom labels.

```python
from transformers import pipeline

classifier = pipeline("zero-shot-classification")
classifier(
    "This is a course about the Transformers library",
    candidate_labels=["education", "politics", "business"],
)
```

Output:
```python
{'sequence': 'This is a course about the Transformers library',
 'labels': ['education', 'business', 'politics'],
 'scores': [0.8445963859558105, 0.111976258456707, 0.043427448719739914]}
```

???
Zero-shot classification is a powerful technique where models can classify text into custom labels without prior training. This is particularly useful when annotating data is time-consuming or requires domain expertise.

---

# Text generation

- Generating text from a prompt using a pipeline.
- Example: Auto-completing a sentence.

```python
from transformers import pipeline

generator = pipeline("text-generation")
generator("In this course, we will teach you how to")
```

Output:
```python
[{'generated_text': 'In this course, we will teach you how to...'}]
```

???
Text generation involves providing a prompt and letting the model auto-complete the text. This is similar to predictive text features on smartphones. The output can vary due to the randomness involved in the process.

---

# Using any model from the Hub in a pipeline

- Selecting specific models from the Hugging Face Model Hub.
- Example: Using a custom model for text generation.

```python
from transformers import pipeline

generator = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-360M")
generator(
    "In this course, we will teach you how to",
    max_length=30,
    num_return_sequences=2,
)
```

Output:
```python
[{'generated_text': '...'}, {'generated_text': '...'}]
```

???
You can use any model from the Hugging Face Model Hub in a pipeline. This flexibility allows you to choose models that best fit your specific task, such as text generation in different languages.

---

# Mask filling

- Filling in missing words in text using a pipeline.
- Example: Predicting missing words in a sentence.

```python
from transformers import pipeline

unmasker = pipeline("fill-mask")
unmasker("This course will teach you all about <mask> models.", top_k=2)
```

Output:
```python
[{'sequence': '...', 'score': ..., 'token': ..., 'token_str': '...'},
 {'sequence': '...', 'score': ..., 'token': ..., 'token_str': '...'}]
```

???
Mask filling involves predicting missing words in a sentence. The model fills in the special `<mask>` token, and you can control how many predictions are displayed using the `top_k` argument.

---

# Named entity recognition

- Identifying entities like persons, locations, organizations.
- Example: Extracting entities from a sentence.

```python
from transformers import pipeline

ner = pipeline("ner", grouped_entities=True)
ner("My name is Sylvain and I work at Hugging Face in Brooklyn.")
```

Output:
```python
[{'entity_group': 'PER', 'score': ..., 'word': 'Sylvain', 'start': ..., 'end': ...},
 {'entity_group': 'ORG', 'score': ..., 'word': 'Hugging Face', 'start': ..., 'end': ...},
 {'entity_group': 'LOC', 'score': ..., 'word': 'Brooklyn', 'start': ..., 'end': ...}]
```

???
Named entity recognition (NER) involves identifying and classifying entities in text, such as persons, locations, and organizations. The pipeline can group together parts of the sentence that correspond to the same entity.

---

# Question answering

- Answering questions based on a given context.
- Example: Extracting an answer from a context.

```python
from transformers import pipeline

question_answerer = pipeline("question-answering")
question_answerer(
    question="Where do I work?",
    context="My name is Sylvain and I work at Hugging Face in Brooklyn",
)
```

Output:
```python
{'score': ..., 'start': ..., 'end': ..., 'answer': 'Hugging Face'}
```

???
The question-answering pipeline answers questions by extracting information from a provided context. It does not generate answers but rather identifies the relevant portion of the text.

---

# Summarization

- Creating a shorter version of a text while preserving key information.
- Example: Summarizing a long text.

```python
from transformers import pipeline

summarizer = pipeline("summarization")
summarizer("..."))
```

Output:
```python
[{'summary_text': '...'}]
```

???
Summarization involves reducing a text into a shorter version while retaining the most important information. This is useful for quickly understanding the main points of a lengthy document.

---

# Translation

- Translating text from one language to another.
- Example: Translating French to English.

```python
from transformers import pipeline

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-fr-en")
translator("Ce cours est produit par Hugging Face.")
```

Output:
```python
[{'translation_text': 'This course is produced by Hugging Face.'}]
```

???
Translation pipelines can translate text between languages. You can specify the language pair or choose a model from the Hugging Face Model Hub for more flexibility.

---

# Image and audio pipelines

- Image classification: Identifying objects in images.
- Automatic speech recognition: Converting speech to text.

```python
from transformers import pipeline

image_classifier = pipeline("image-classification")
result = image_classifier("...")
print(result)
```

Output:
```python
[...]
```

???
Transformer models are not limited to text; they can also process images and audio. Image classification identifies objects in images, while automatic speech recognition converts spoken language into text.

---

# Combining data from multiple sources

- Searching across multiple databases or repositories.
- Consolidating information from different formats.
- Creating a unified view of related information.

???
One of the powerful applications of Transformer models is their ability to combine and process data from multiple sources. This is useful for tasks like searching across databases, consolidating information, and creating unified views of related data.

---

# Conclusion

- Pipelines are designed for specific tasks and cannot perform variations.
- Next chapter: Customizing pipeline behavior and understanding their internals.

???
The pipelines demonstrated in this chapter are primarily for illustrative purposes and are tailored to specific tasks. In the next chapter, we'll delve into the internals of pipelines and learn how to customize their behavior for more advanced use cases.
```