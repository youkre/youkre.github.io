---
title: "第6章 向量语义与词嵌入"
summary: ""
date: 
---

> 荃者所以在鱼，得鱼而忘荃 Nets are for ﬁsh;
> Once you get the ﬁsh, you can forget the net.
> 言者所以在意，得意而忘言 Words are for meaning;
> Once you get the meaning, you can forget the words
> -- 庄子(Zhuangzi), Chapter 26

洛杉矶以其遍布的沥青路面闻名，这些路面主要集中在高速公路上。但在城市中心还有另一片沥青区域——拉布雷亚沥青坑（La Brea tar pits），这里保存了数百万块来自更新世冰河时代末期的化石骨骼。其中一种化石就是刃齿虎（Smilodon），或称剑齿虎，其标志性的长犬齿令人过目难忘。大约五百万年前，在阿根廷和南美洲其他地区生活着另一种完全不同的剑齿虎——袋剑虎（Thylacosmilus）。袋剑虎是有袋类动物，而刃齿虎是胎盘哺乳动物，但袋剑虎也拥有长长的上犬齿，并且和刃齿虎一样，下颌具有保护性的骨质突缘。这两种哺乳动物的相似性是“平行进化”或“趋同进化”的众多例子之一：特定的环境或背景会导致不同物种演化出非常相似的结构（Gould, 1980）。

**语境**（context）的作用在另一种不那么生物性的“有机体”——词语——的相似性中也同样重要。出现在相似语境中的词语往往具有相似的含义。这种词语分布上的相似性与词义相似性之间的联系被称为**分布假说**（distributional hypothesis）。这一假说最早由20世纪50年代的语言学家如乔斯（Joos, 1950）、哈里斯（Harris, 1954）和弗斯（Firth, 1957）提出。他们注意到，同义词（如 oculist 和 eye-doctor）往往出现在相同的环境中（例如，靠近 eye 或 examined 等词），且两个词之间的语义差异“大致对应于它们环境中的差异程度”（Harris, 1954, 第157页）。

在本章中，我们将介绍**向量语义**（vector semantics），它通过直接从文本中的分布情况学习词语意义的表示——即**嵌入**（embeddings）——来实现这一语言学假说。这些表示被应用于所有涉及语义的自然语言处理任务中，我们在此介绍的**静态嵌入**（static embeddings）构成了更强大的动态**或上下文化嵌入**（contextualized embeddings）的基础，例如我们将在第11章中介绍的**BERT**。

这些词语表示也是本书中出现的第一个**表示学习**（representation learning）的例子，即自动学习输入文本的有用表示。研究如何找到这类**自监督**（self-supervised）的方式来学习输入表示，而不是通过**特征工程**（feature engineering）手动创建表示，是当前NLP研究的一个重要方向（Bengio et al., 2013）。


<nav class="pagination justify-content-between">
<a href="../ch5-11">5.11 小结</a>
<a href="../">目录</a>
<a href="../ch6-01">6.1 词汇语义</a>
</nav>

