In this video, we're going to go over the HuggingFace Model Hub navigation.

This is the huggingface.co landing page. To access the model hub, click on the "Models" tab in the upper right corner.

You should be facing this web interface, which can be split into several parts.

On the left, you'll find categories, which you can use to tailor your model search.

The first category is the "Tasks". Models on the hub may be used for a wide variety of tasks. These include natural language processing tasks, such as question answering or text classification, but it isn't only limited to NLP. Other tasks from other fields are also available, such as image classification for computer vision, or automatic speech recognition for speech.

The second category is the "libraries". Models on the hub usually share one of three backbones: PyTorch, TensorFlow, or JAX. However, other backbones, such as rust or ONNX also exist.

Finally, this tab can also be used to specify from which high-level framework the model comes. This includes Transformers, but it isn't limited to it. The model Hub is used to host a lot of different frameworks' models, and we are actively looking to host other frameworks' models.

The third category is the "Datasets" tab. Selecting a dataset from this tab means filtering the models so that they were trained on that specific dataset.

The fourth category is the "Languages" tab. Selecting a language from this tab means filtering the models so that they handle the language selected.

Finally, the last category allows to choose the license with which the model is shared.

On the right, you'll find the models available on the model Hub!

The models are ordered by downloads. When clicking on a model, you should be facing its model card. The model card contains information about the model: its description, intended use, limitations and biases. It can also show code snippets on how to use the model, as well as any relevant information: training procedure, data processing, evaluation results, copyrights.

This information is crucial for the model to be used. The better crafted a model card is, the easier it will be for other users to leverage your model in their applications.

On the right of the model card is the inference API. This inference API can be used to play with the model directly. Feel free to modify the text and click on compute to see how would the model behave to your inputs.

At the top of the screen lie the model tags. These include the model task, as well as any other tag that is relevant to the categories we have just seen.

The "Files & Versions tab" displays the architecture of the repository of that model. Here, we can see all the files that define this model. You'll see all usual features of a git repository: the branches available, the commit history as well as the commit diff.

Three different buttons are available at the top of the model card. The first one shows how to use the inference API programmatically. The second one shows how to train this model in SageMaker, and the last one shows how to load that model within the appropriate library. For BERT, this is transformers.