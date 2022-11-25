Let's see together what is the training strategy of the WordPieces algorithm and how it performs the tokenization of a text once trained.

WordPieces is a tokenization algorithm introduced by Google. It is used for example by Bert. To our knowledge, the code of Word Pieces has not been open sourced, so we base our explanations on our own interpretation of the published literature.

What is the training strategy of WordPieces? Similarly to the BPE algorithm, WordPieces starts by establishing an initial vocabulary composed of elementary units and then increases this vocabulary to the desired size. To build the initial vocabulary, we divide each word in the training corpus into the sequence of letters that make it up. As you can see, there is a small subtlety: we add a 2 hashtags in front of the letters that do not start a word. By keeping only one occurrence per elementary unit we now have our initial vocabulary.

We will list all the existing pairs in our corpus. Once we have this list, we will calculate a score for each of these pairs. As for the BPE algorithm, we will select the pair with the highest score.

Taking for example the first pair composed of H and U. The score of a pair is simply equal to the frequency of appearance of the pair divided by the product of the frequency of appearance of the first token by the frequency of appearance of the second token. Thus at a fixed frequency of appearance of the pair, if the subparts of the pair are very frequent in the corpus then this score will be decreased. In our example, the pair "hu" appears 4 times, the letter "h" 4 times and the letter u 4 times. This gives us a score of 0.25.

Now that we know how to calculate this score, we can do it for all pairs. We can now add to the vocabulary the pair with the highest score, after merging it of course! And now we can apply this same fusion to our split corpus.

As you can imagine, we just have to repeat the same operations until we have the vocabulary at the desired size! Let's look at a few more steps to see the evolution of our vocabulary and the length of the splits getting shorter.
Now that we are happy with our vocabulary, you are probably wondering how to use it to tokenize a text.  Let's say we want to tokenize the word "huggingface".  WordPieces follows these rules: We will look for the longest possible token at the beginning of our word. Then we start again on the remaining part of our word. And so on until we reach the end! And that's it, huggingface is divided into 4 sub-tokens.
This video is about to end, I hope it helped you to understand better what is behind the word WordPieces!