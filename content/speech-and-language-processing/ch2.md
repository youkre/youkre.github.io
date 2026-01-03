---
title: "第2章 词与词元"
summary: ""
date: 2025-12-20T20:08:00+08:00
---

> User: I need some help, that much seems certain.  
> ELIZA: WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP  
> User: Perhaps I could learn to get along with my mother.  
> ELIZA: TELL ME MORE ABOUT YOUR FAMILY  
> User: My mother takes care of me.  
> ELIZA: WHO ELSE IN YOU FAMILY TAKES CARE OF YOU  
> User: My father.  
> ELIZA: YOUR FATHER  
> User: You are like my father in some ways.  
> Weizenbaum (1966)

上述对话来自 **ELIZA**——一个早期的自然语言处理系统。该系统通过模仿罗杰斯学派心理治疗师（Rogerian psychotherapist）的回应方式，能够与用户进行有限的对话（Weizenbaum, 1066）。
ELIZA 的实现令人惊讶地简单：它通过对词（words）进行模式匹配，识别诸如 “I need X” 这样的短语，并将其转换为合适的输出，例如 “What would it mean to you if you got X?”。
尽管以现代标准来看，ELIZA 对人类对话的模拟非常粗糙，但其效果却出奇地成功：许多与 ELIZA 交互的用户竟相信它真正“理解”了自己。
正因如此，这项工作首次促使研究者开始思考聊天机器人对其用户可能产生的影响（Weizenbaum, 1976）。

当然，ELIZA 所开创的那种基于模式的简单模仿，现代聊天机器人已不再使用。
然而，ELIZA 所体现的这种基于模式的词处理方法，在当今的词元化（tokenization）任务中依然适用；词元化即指从连续文本中将词及其组成部分转化为词元的过程。
词元化是现代自然语言处理（NLP）的第一步，采用了一些源于 ELIZA 时代的基于模式的方法。

要理解词元化，我们首先需要提问：什么是词（word）？
“um” 算是一个词吗？“New York” 呢？
不同语言中“词”的本质是否相似？
有些语言（如越南语或粤语）的词通常非常短，而另一些语言（如土耳其语）的词则非常长。
此外，我们还需考虑如何用**字符**（characters）来表示词。
为此，我们将介绍 **Unicode**——现代字符表示系统，以及 **UTF-8** 文本编码方案。
同时，我们还将引入 **语素**（morpheme）的概念，即词中有意义的子成分（例如单词 `longer` 中的语素 `-er`）。

文本词元化的标准方法是利用输入字符提供切分线索。
因此，在理解了词可能包含的子成分之后，我们将介绍标准的 **字节对编码**（Byte-Pair Encoding, **BPE**）算法，该算法能自动将输入文本切分为词元（tokens）。
BPE 利用字符序列的简单统计规律，归纳出一个子词（subword）词元的词汇表。
所有词元化系统在处理过程中也都依赖于 **正则表达式**（regular expressions）。
正则表达式是一种用于形式化描述和操作文本字符串的语言，是所有现代 NLP 系统中的重要工具。
我们将介绍正则表达式，并展示其应用示例。

最后，我们将引入一种名为 **编辑距离**（edit distance）的指标，用于衡量两个词或字符串之间的相似程度，其依据是将一个字符串转换为另一个所需执行的编辑操作次数（包括插入、删除和替换）。
在 NLP 中，只要需要比较两个词或字符串，编辑距离就会发挥作用，例如在自动语音识别中至关重要的 **词错误率**（word error rate, WER）指标中。


<nav class="pagination justify-content-between">
<span></span>
<a href="../">目录</a>
<a href="../ch2-01">2.1 词</a>
</nav>

