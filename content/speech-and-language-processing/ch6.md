---
title: "CHAPTER 6 Vector Semantics and Embeddings"
summary: ""
date: 
---

> 荃者所以在鱼，得鱼而忘荃 Nets are for ﬁsh;
> Once you get the ﬁsh, you can forget the net.
> 言者所以在意，得意而忘言 Words are for meaning;
> Once you get the meaning, you can forget the words
> -- 庄子(Zhuangzi), Chapter 26

The asphalt that Los Angeles is famous for occurs mainly on its freeways. But
in the middle of the city is another patch of asphalt, the La Brea tar pits, and this
asphalt preserves millions of fossil bones from the last of the Ice Ages of the Pleis-
tocene Epoch. One of these fossils is the Smilodon, or saber-toothed tiger, instantly
recognizable by its long canines. Five million years ago or so, a completely different
saber-tooth tiger called Thylacosmilus lived
in Argentina and other parts of South Amer-
ica. Thylacosmilus was a marsupial whereas
Smilodon was a placental mammal, but Thy-
lacosmilus had the same long upper canines
and, like Smilodon, had a protective bone
ﬂange on the lower jaw. The similarity of
these two mammals is one of many examples
of parallel or convergent evolution, in which particular contexts or environments
lead to the evolution of very similar structures in different species (Gould, 1980).

The role of context is also important in the similarity of a less biological kind
of organism: the word. Words that occur in similar contexts tend to have similar
meanings. This link between similarity in how words are distributed and similarity
in what they mean is called the **distributional hypothesis**. The hypothesis was
ﬁrst formulated in the 1950s by linguists like Joos (1950), Harris (1954), and Firth
(1957), who noticed that words which are synonyms (like oculist and eye-doctor)
tended to occur in the same environment (e.g., near words like eye or examined)
with the amount of meaning difference between two words “corresponding roughly
to the amount of difference in their environments” (Harris, 1954, p. 157).

In this chapter we introduce **vector semantics**, which instantiates this linguistic
hypothesis by learning representations of the meaning of words, called **embeddings**,
directly from their distributions in texts. These representations are used in every nat-
ural language processing application that makes use of meaning, and the **static embeddings**
we introduce here underlie the more powerful dynamic **orcontextualized embeddings**
like **BERT** that we will see in Chapter 11.

These word representations are also the ﬁrst example in this book of repre-
**sentation learning**, automatically learning useful representations of the input text.
Finding such **self-supervised** ways to learn representations of the input, instead of
creating representations by hand via **feature engineering**, is an important focus of
NLP research (Bengio et al., 2013).


<nav class="pagination justify-content-between">
<a href="../ch5-11">5.11 小结</a>
<a href="../">目录</a>
<a href="../ch6-01">6.1 Lexical Semantics</a>
</nav>

