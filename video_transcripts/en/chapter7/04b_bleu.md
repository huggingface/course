What is the BLEU metric? For many NLP tasks we can use common metrics like accuracy or F1 score, but what do you do when you want to measure the quality of text that's generated from a model like GPT-2? In this video, we'll take a look at a widely used metric for machine translation called BLEU, which is short for BiLingual Evaluation Understudy

The basic idea behind BLEU is to assign a single numerical score to a translation that tells us how "good" it is compared to one or more reference translations. In this example we have a sentence in Spanish that has been translated into English by some model. If we compare the generated translation to some reference human translations, we can see that the model is pretty good, but has made a common error: the Spanish word "tengo" means "have" in English and this 1-1 translation is not quite natural.

So how can we measure the quality of a generated translation in an automatic way? The approach that BLEU takes is to compare the n-grams of the generated translation to the n-grams of the references. An n-gram is just a fancy way of saying "a chunk of n words", so let's start with unigrams, which correspond to the individual words in a sentence. In this example you can see that four of the words in the generated translation are also found in one of the reference translations.

Now that we've found our matches, one way to assign a score to the translation is to compute the precision of the unigrams. This means we just count the number of matching words in the generated and reference translations and normalize the count by dividing by the number of word in the generation. In this example, we found 4 matching words and our generation has 5 words, so our unigram precision is 4/5 or 0.8. In general precision ranges from 0 to 1, and higher precision scores mean a better translation.

One problem with unigram precision is that translation models sometimes get stuck in repetitive patterns and repeat the same word several times. If we just count the number of word matches, we can get really high precision scores even though the translation is terrible from a human perspective! For example, if our model just generates the word "six", we get a perfect unigram precision score.

To handle this, BLEU uses a modified precision that clips the number of times to count a word, based on the maximum number of times it appears in the reference translation. In this example, the word "six" only appears once in the reference, so we clip the numerator to one and the modified unigram precision now gives a much lower score.

Another problem with unigram precision is that it doesn't take into account the order of the words in the translations. For example, suppose we had Yoda translate our Spanish sentence, then we might get something backwards like "years six thirty have I". In this case, the modified unigram precision gives a high precision which is not what we want.

So to deal with word ordering problems, BLEU actually computes the precision for several different n-grams and then averages the result. For example, if we compare 4-grams, then we can see there are no matching chunks of 4 words in translations and so the 4-gram precision is 0.

To compute BLEU scores in 🤗 Datasets is very simple: just use the load_metric() function, provide your model's predictions along with the references and you're good to go!

The output contains several fields of interest. The precisions field contains all the individual precision scores for each n-gram

The BLEU score itself is then calculated by taking the geometric mean of the precision scores. By default, the mean of all four n-gram precisions is reported, a metric that is sometimes also called BLEU-4. In this example we can see the BLEU score is zero because the 4-gram precision was zero.

The BLEU metric has some nice properties, but it is far from a perfect metric. The good properties are that it's easy to compute and widely used in research so you can compare your model against others on a benchmark. On the other hand, there are several problems with BLEU, including the fact it doesn't incorporate semantics and struggles on non-English languages. Another problem with BLEU is that it assumes the human translations have already been tokenized and this makes it hard to compare models with different tokenizers.

Measuring the quality of texts is still a difficult, open problem in NLP research. For machine translation, the current recommendation is to use the SacreBLEU metric which addresses the tokenization limitations of BLEU. As you can see in this example, computing the SacreBLEU score is almost identical to the BLEU one. The main difference is that we now pass a list of texts instead of a list of words for the translations, and SacreBLEU takes care of the tokenization under the hood.