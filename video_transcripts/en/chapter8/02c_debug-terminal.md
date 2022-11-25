Using the Python debugger in a terminal. In this video, we'll learn how to use the Python debugger in a terminal. For this example, we are running code from the token classification section, downloading the Conll dataset before loading a tokenizer to preprocess it. Checkout the section of the course linked below for more information.

Once this is done, we try to batch together some features of the training dataset by padding them and returning a tensor, then we get the following error. We use PyTorch here but you will get the same error with TensorFlow. As we have seen in the "How to debug an error?" video, the error message is at the end and it indicates we should use padding... which we are actually trying to do.

So this is not useful and we will need to go a little deeper to debug the problem. Fortunately, you can use the Python debugger quite easily in a terminal by launching your script with python -m pdb instead of just python.

When executing that command, you are sent to the first instruction of your script. You can run just the next instruction by typing n, or continue to the error by directly typing c. Once there, you go to the very bottom of the traceback, and you can type commands. The first two commands you should learn are u and d (for up and down), which allow you to go up in the Traceback or down. Going up twice, we get to the point the error was reached.

The third command to learn is p, for print. It allows you to print any value you want. For instance here, we can see the value of return_tensors or batch_outputs to try to understand what triggered the error.

The batch outputs dictionary is a bit hard to see, so let's dive into smaller pieces of it. Inside the debugger you can not only print any variable but also evaluate any expression, so we can look independently at the inputs or labels.

Those labels are definitely weird: they are of various size, which we can actually confirm by printing the sizes. No wonder the tokenizer wasn't able to create a tensor with them! This is because the pad method only takes care of the tokenizer outptus: input IDs, attention mask and token type IDs, so we have to pad the labels ourselves before trying to create a tensor with them.

Once you are ready to exit the Python debugger, you can press q for quit.

Another way we can access the Python debugger is to set a "set_trace" instruction where we want in the script. It will interrupt the execution and launch the Python debugger at this place, and we can inspect all the variables before the next instruction is executed. Typing n executes the next instruction, which takes us back inside the traceback.

One way to fix the error is to manually pad all labels to the longest, or we can use the data collator designed for this.