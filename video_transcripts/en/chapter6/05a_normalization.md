In this video we will see together what is the normalizer component that we find at the beginning of each tokenizer.

The normalization operation consists in applying a succession of normalization rules to the raw text. We choose normalization rules to remove noise in the text which seems useless for the learning and use of our language model.

Let's take a very diverse sentence with different fonts, upper and lower case characters, accents, punctuation and multiple spaces, to see how several tokenizers normalize it. The tokenizer from the FNet model has transformed the letters with font variants or circled into their basic version and has removed the multiple spaces. And now if we look at the normalization with Retribert's tokenizer, we can see that it keeps characters with several font variants and keeps the multiple spaces but it removes all the accents. And if we continue to test the normalization of many other tokenizers associated to models that you can find on the Hub we can see that they also propose other normalizations.

With the fast tokenizers, it is very easy to observe the normalization chosen for the currently loaded tokenizer. Indeed, each instance of a fast tokenizer has an underlying tokenizer from the Tokenizers library stored in the backend_tokenizer attribute. This object has itself a normalizer attribute that we can use thanks to the "normalize_str" method to normalize a string.

It is thus very practical that this normalization which was used at the time of the training of the tokenizer was saved and that it applies automatically when you asks a trained tokenizer to tokenize a text. For example, if we hadn't included the albert normalizer we would have had a lot of unknown tokens by tokenizing this sentence with accents and capital letters.

These transformations can also be undetectable with a simple "print". Indeed, keep in mind that for a computer, text is only a succession of 0 and 1 and it happens that different successions of 0 and 1 render the same printed character. 

the 0s and 1s go in groups of 8 to form a byte. The computer must then decode this sequence of bytes into a sequence of "code points". In our example the 2 bytes are transformed into a single "code point" by UTF-8. The unicode standard then allows us to find the character corresponding to this  code point: the c cedilla. Let's repeat the same operation with this new sequence composed of 3 bytes, this time it is transformed into 2 "code points" .... which also correspond to the c cedilla character! It is in fact the composition of the unicode [Latin Small Letter C](https://unicode-table.com/en/0063/) and the [combining cedilla](https://unicode-table.com/en/0327/). But it's annoying because what appears to us to be a single character is not at all the same thing for the computer. Fortunately, there are unicode standardization standards known as NFC, NFD, NFKC and NFKD that allow erasing some of these differences. These standards are often used by tokenizers!

On all these previous examples, even if the normalizations changed the look of the text, they did not change the content: you could still read "Hello world, let's normalize this sentence". However, you must be aware that some normalizations can be very harmful if they are not adapted to their corpus. For example, if you take the French sentence "un père indigné", which means "An indignant father", and normalize it with the bert-base-uncase tokenizer which removes the accents then the sentence becomes "un père indigne" which means "An unworthy father". If you watch this video to build your own tokenizer, there are no absolute rules to choose or not a normalization for your brand new tokenizer but I advise you to take the time to select them so that they do not make you lose important information.