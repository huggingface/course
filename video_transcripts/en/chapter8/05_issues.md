How to write a good issue on GitHub?

GitHub is the main place for the Hugging Face opensource libraries, and should always go there to report a bug or ask for a new feature. For more general questions or to debug your own code, use the forums (see the video linked below). It's very important to write good issues as it will help the bug you uncovered be fixed in no time.

For this video, we have created a version of Transformers with a bug. You can install it by executing this command in a notebook (remove the exclamation mark to execute it in a terminal). In this version, the following example fails.

The error is rather cryptic and does not seem to come from anything in our code, so it seems we have a bug to report!

The first thing to do in this case is to try to find the smallest amount of code possible that reproduces the bug. In our case, inspecting the traceback, we see the failure happens inside the pipeline function when it calls AutoTokenizer.from_pretrained.

Using the debugger, we find the values passed to that method and can thus create a small sample of code that hopefully generates the same error.

It's very important to go though this step as you may realize the error was on your side and not a bug in the library, but it also will make it easier for the maintainers to fix your problem. Here we can play around a bit more with this code and notice the error happens for different checkpoints and not just this one, and that it disappears when we use use_fast=False inside our tokenizer call. The important part is to have something that does not depend on any external files or data. Try to replace your data by fake values if you can't share it.

With all of this done, we are ready to start writing our issue. () Click on the button next to Bug Report

and you will discover there is a template to fill. It will only take you a couple of minutes.

The first thing is to properly name your issue. Don't pick a title that is too vague!

Then you have to fill your environment information. There is a command provided by the Transformers library to do this.

Just execute it in your notebook or in a terminal, and copy paste the results.

There are two last questions to fill manually (to which the answers are no and no in our case).

Next, we need to determine who to tag. There is a full list of usernames. Since our issue has to do with tokenizers, () we pick the maintainer associated with them. There is no point tagging more than 3 people, they will redirect you to the right person if you made a mistake.

Next, we have to give the information necessary to reproduce the bug. We paste our sample, and put it between () two lines with three backticks so it's formatted properly.

We also paste the full traceback, still between two lines of three backticks.

Lastly, we can add any additional information about what we tried to debug the issue at hand. With all of this, you should expect an answer to your issue pretty fast, and hopefully, a quick fix! Note that all the advise in this video applies for almost every open-source project.