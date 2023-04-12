# Asking for help on the forums[[asking-for-help-on-the-forums]]

<CourseFloatingBanner chapter={8}
  classNames="absolute z-10 right-0 top-0"
  notebooks={[
    {label: "Google Colab", value: "https://colab.research.google.com/github/huggingface/notebooks/blob/master/course/en/chapter8/section3.ipynb"},
    {label: "Aws Studio", value: "https://studiolab.sagemaker.aws/import/github/huggingface/notebooks/blob/master/course/en/chapter8/section3.ipynb"},
]} />

<Youtube id="S2EEG3JIt2A"/>

The [Hugging Face forums](https://discuss.huggingface.co) are a great place to get help from the open source team and wider Hugging Face community. Here's what the main page looks like on any given day:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forums.png" alt="The Hugging Face forums." width="100%"/>
</div>

On the lefthand side you can see all the categories that the various topics are grouped into, while the righthand side shows the most recent topics. A topic is a post that contains a title, category, and description; it's quite similar to the GitHub issues format that we saw when creating our own dataset in [Chapter 5](/course/chapter5). As the name suggests, the [Beginners](https://discuss.huggingface.co/c/beginners/5) category is primarily intended for people just starting out with the Hugging Face libraries and ecosystem. Any question on any of the libraries is welcome there, be it to debug some code or to ask for help about how to do something. (That said, if your question concerns one library in particular, you should probably head to the corresponding library category on the forum.)

Similarly, the [Intermediate](https://discuss.huggingface.co/c/intermediate/6) and [Research](https://discuss.huggingface.co/c/research/7) categories are for more advanced questions, for example about the libraries or some cool new NLP research that you'd like to discuss.

And naturally, we should also mention the [Course](https://discuss.huggingface.co/c/course/20) category, where you can ask any questions you have that are related to the Hugging Face course!

Once you have selected a category, you'll be ready to write your first topic. You can find some [guidelines](https://discuss.huggingface.co/t/how-to-request-support/3128) in the forum on how to do this, and in this section we'll take a look at some features that make up a good topic.

## Writing a good forum post[[writing-a-good-forum-post]]

As a running example, let's suppose that we're trying to generate embeddings from Wikipedia articles to create a custom search engine. As usual, we load the tokenizer and model as follows:

```python
from transformers import AutoTokenizer, AutoModel

model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)
```

Now suppose we try to embed a whole section of the [Wikipedia article](https://en.wikipedia.org/wiki/Transformers) on Transformers (the franchise, not the library!):

```python
text = """
Generation One is a retroactive term for the Transformers characters that
appeared between 1984 and 1993. The Transformers began with the 1980s Japanese
toy lines Micro Change and Diaclone. They presented robots able to transform
into everyday vehicles, electronic items or weapons. Hasbro bought the Micro
Change and Diaclone toys, and partnered with Takara. Marvel Comics was hired by
Hasbro to create the backstory; editor-in-chief Jim Shooter wrote an overall
story, and gave the task of creating the characthers to writer Dennis O'Neil.
Unhappy with O'Neil's work (although O'Neil created the name "Optimus Prime"),
Shooter chose Bob Budiansky to create the characters.

The Transformers mecha were largely designed by Shōji Kawamori, the creator of
the Japanese mecha anime franchise Macross (which was adapted into the Robotech
franchise in North America). Kawamori came up with the idea of transforming
mechs while working on the Diaclone and Macross franchises in the early 1980s
(such as the VF-1 Valkyrie in Macross and Robotech), with his Diaclone mechs
later providing the basis for Transformers.

The primary concept of Generation One is that the heroic Optimus Prime, the
villainous Megatron, and their finest soldiers crash land on pre-historic Earth
in the Ark and the Nemesis before awakening in 1985, Cybertron hurtling through
the Neutral zone as an effect of the war. The Marvel comic was originally part
of the main Marvel Universe, with appearances from Spider-Man and Nick Fury,
plus some cameos, as well as a visit to the Savage Land.

The Transformers TV series began around the same time. Produced by Sunbow
Productions and Marvel Productions, later Hasbro Productions, from the start it
contradicted Budiansky's backstories. The TV series shows the Autobots looking
for new energy sources, and crash landing as the Decepticons attack. Marvel
interpreted the Autobots as destroying a rogue asteroid approaching Cybertron.
Shockwave is loyal to Megatron in the TV series, keeping Cybertron in a
stalemate during his absence, but in the comic book he attempts to take command
of the Decepticons. The TV series would also differ wildly from the origins
Budiansky had created for the Dinobots, the Decepticon turned Autobot Jetfire
(known as Skyfire on TV), the Constructicons (who combine to form
Devastator),[19][20] and Omega Supreme. The Marvel comic establishes early on
that Prime wields the Creation Matrix, which gives life to machines. In the
second season, the two-part episode The Key to Vector Sigma introduced the
ancient Vector Sigma computer, which served the same original purpose as the
Creation Matrix (giving life to Transformers), and its guardian Alpha Trion.
"""

inputs = tokenizer(text, return_tensors="pt")
logits = model(**inputs).logits
```

```python output
IndexError: index out of range in self
```

Uh-oh, we've hit a problem -- and the error message is far more cryptic than the ones we saw in [section 2](/course/chapter8/section2)! We can't make head or tails of the full traceback, so we decide to turn to the Hugging Face forums for help. How might we craft the topic?

To get started, we need to click the "New Topic" button at the upper-right corner (note that to create a topic, we'll need to be logged in):

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forums-new-topic.png" alt="Creating a new forum topic." width="100%"/>
</div>

This brings up a writing interface where we can input the title of our topic, select a category, and draft the content:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic01.png" alt="The interface for creating a forum topic." width="100%"/>
</div>

Since the error seems to be exclusively about 🤗 Transformers, we'll select this for the category. Our first attempt at explaining the problem might look something like this:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic02.png" alt="Drafting the content for a new forum topic." width="100%"/>
</div>

Although this topic contains the error message we need help with, there are a few problems with the way it is written:

1. The title is not very descriptive, so anyone browsing the forum won't be able to tell what the topic is about without reading the body as well.
2. The body doesn't provide enough information about _where_ the error is coming from and _how_ to reproduce it.
3. The topic tags a few people directly with a somewhat demanding tone.

Topics like this one are not likely to get a fast answer (if they get one at all), so let's look at how we can improve it. We'll start with the first issue of picking a good title.

### Choosing a descriptive title[[choosing-a-descriptive-title]]

If you're trying to get help with a bug in your code, a good rule of thumb is to include enough information in the title so that others can quickly determine whether they think they can answer your question or not. In our running example, we know the name of the exception that's being raised and have some hints that it's triggered in the forward pass of the model, where we call `model(**inputs)`. To communicate this, one possible title could be:

> Source of IndexError in the AutoModel forward pass?

This title tells the reader _where_ you think the bug is coming from, and if they've encountered an `IndexError` before, there's a good chance they'll know how to debug it. Of course, the title can be anything you want, and other variations like:

> Why does my model produce an IndexError?

could also be fine. Now that we've got a descriptive title, let's take a look at improving the body.

### Formatting your code snippets[[formatting-your-code-snippets]]

Reading source code is hard enough in an IDE, but it's even harder when the code is copied and pasted as plain text! Fortunately, the Hugging Face forums support the use of Markdown, so you should always enclose your code blocks with three backticks (```) so it's more easily readable. Let's do this to prettify the error message -- and while we're at it, let's make the body a bit more polite than our original version:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic03.png" alt="Our revised forum topic, with proper code formatting." width="100%"/>
</div>

As you can see in the screenshot, enclosing the code blocks in backticks converts the raw text into formatted code, complete with color styling! Also note that single backticks can be used to format inline variables, like we've done for `distilbert-base-uncased`. This topic is looking much better, and with a bit of luck we might find someone in the community who can guess what the error is about. However, instead of relying on luck, let's make life easier by including the traceback in its full gory detail!

### Including the full traceback[[including-the-full-traceback]]

Since the last line of the traceback is often enough to debug your own code, it can be tempting to just provide that in your topic to "save space." Although well intentioned, this actually makes it _harder_ for others to debug the problem since the information that's higher up in the traceback can be really useful too. So, a good practice is to copy and paste the _whole_ traceback, while making sure that it's nicely formatted. Since these tracebacks can get rather long, some people prefer to show them after they've explained the source code. Let's do this. Now, our forum topic looks like the following:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic04.png" alt="Our example forum topic, with the complete traceback." width="100%"/>
</div>

This is much more informative, and a careful reader might be able to point out that the problem seems to be due to passing a long input because of this line in the traceback:

> Token indices sequence length is longer than the specified maximum sequence length for this model (583 > 512).

However, we can make things even easier for them by providing the actual code that triggered the error. Let's do that now.

### Providing a reproducible example[[providing-a-reproducible-example]]

If you've ever tried to debug someone else's code, you've probably first tried to recreate the problem they've reported so you can start working your way through the traceback to pinpoint the error. It's no different when it comes to getting (or giving) assistance on the forums, so it really helps if you can provide a small example that reproduces the error. Half the time, simply walking through this exercise will help you figure out what's going wrong. In any case, the missing piece of our example is to show the _inputs_ that we provided to the model. Doing that gives us something like the following completed example:

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface-course/documentation-images/resolve/main/en/chapter8/forum-topic05.png" alt="The final version of our forum topic." width="100%"/>
</div>

This topic now contains quite a lot of information, and it's written in a way that is much more likely to attract the attention of the community and get a helpful answer. With these basic guidelines, you can now create great topics to find the answers to your 🤗 Transformers questions!

