In this video, we're going to understand how to manage a model repository on the HuggingFace model hub. In order to handle a repository, you should first have a Hugging Face account.

A link to create a new account is available in the description.

Once you are logged in, you can create a new repository by clicking on the "New model" option.

You should be facing a similar modal to the following:

- In the "Owner" input, you can put either your own namespace or any of your organisations namespaces.
- The model name is the model identifier that will then be used to identify your model on your chosen namespace.
- The final choice is between public and private:
    - Public models are accessible by anyone. This is the recommended, free option, as this makes your model easily accessible and shareable. The owners of your namespace are the only ones who can update and change your model.
    - A more advanced option is the private option. In this case, only the owners of your namespace will have visibility over your model. Other users won't know it exists and will not be able to use it.

Let's create a dummy model to play with.

Once your model is created, comes the management of that model! Three tabs are available to you. You're facing the first one, which is the model card page; this is the page used to showcase your model to the world. We'll see how it can be completed in a bit.

The second one is the "Files & Versions". Your model itself is a git repository - if you're unaware of what is a git repository, you can think of it as a folder containing files. If you have never used git before, we recommend looking at an introduction like the one provided in this video's description. The git repository allows you to see the changes happening over time in this folder, hence the term "Versions". We'll see how to add files and versions in a bit.

The final tab is the "Settings" tab, which allow you to manage your model's visibility and availability.

Let's first start by adding files to the repository.

Files can be added through the web interface thanks to the "Add File" button. The added files can be of any type: python, json, text, you name it! Alongside your added file and its content, you should name your change, or commit.

- Generally, adding files is simpler when using the command line. We'll showcase how to do this using git. In addition to git, we're using git-lfs, which stands for large file storage in order to manage large model files.
    - First, I make sure that both git and git-lfs are correctly installed on my system. Links to install git & git-lfs are provided in the video description.
    - Then, we can get to work by cloning the repository locally. We have a repository with a single file! The file that we have just added to the repository using the web interface. We can edit it to see the contents of this file and update these.
    - It just turns out I have a model handy, that can be used for sentiment analysis. I'll simply copy over the contents to this folder. This includes the model weights, configuration file and tokenizer to the repository.
    - I can then track these two files with the git add command.
    - Then, I commit the changes. I'm giving this commit the title of "Add model weights and configuration"
    - Finally, I can push the new commit to the [huggingface.co](http://huggingface.co/) remote.
    - When going back to the files & versions tab, we can now see the newly added commit with the updated files
    - We have seen two ways of adding files to a repository, a third way is explored in the video about the push to hub API. A link to this video is in the description.
- Go back to readme
    - Unfortunately, the front page of our model is still very empty. Let's add a README markdown file to complete it a little bit.
    - This README is known as the modelcard, and it's arguably as important as the model and tokenizer files in a model repository. It is the central definition of the model, ensuring reusability by fellow community members and reproducibility of results, and providing a platform on which other members may build their artifacts.
    - We'll only add a title and a small description here for simplicity's sake, but we encourage you to add information relevant to how was the model trained, its intended uses and limitations, as well as its identified and potential biases,  evaluation results and code samples on how your model should be used.

Great work contributing a model to the model hub! This model can now be used in downstream libraries simply by specifying your model identifier.