In this video, we will learn the first things to do when you get an error.

Let's say we want to use the question answering pipeline on a particular model...

... and we get the following error. Errors in Python can appear overwhelming because you get so much information printed out, but that's because Python is trying to help you the best it can to solve your problem. In this video we will see how to interpret the error report we get.

The first thing to notice at the very top is that Python shows you with a clear arrow the line of code that triggered the error. So you don't have to fiddle with your code and remove random lines to figure out where the error comes from, you have the answer in front right here.

The arrows you see below are the parts of the code Python tried to execute while running the instruction: here we are inside the pipeline function and the error came on this line while trying to execute the function check_tasks, () which then raised the KeyError we see displayed. () Note that Python tells you exactly where the functions it's executing live, so if you feel adventurous, you can even go inspect the source code. This whole thing is called the traceback.

If you are running your code on Colab, the Traceback is automatically minimized, () so you have to click to expand it.

At the very end of the traceback, you finally get the actual error message. The first thing you should do when encountering an error is to read that error message. Here it's telling us it doesn't know the quesiton answering task, and helpfully gives us the list of supported tasks... () in which we can see that question answering is.

looking more closely though, we used an underscore to separate the two words when the task is written with a minus, so we should fix that!

Now let's retry our code with the task properly written...

... and what is happening today? Another error! As we saw before, () we go look at the bottom to read the actual error message. It's telling us that we should check our model is a correct model identifier, so let's hop on to [hf.co/models](http://hf.co/models) 

We can see our model listed there () in the ones available for question answering.

The difference is that it's spelled distilbert with one l, and we used two.

So let's fix that. We finally get our results! If your error is more complex, you might need to use the Python debugger, check out the videos linked below to learn how!