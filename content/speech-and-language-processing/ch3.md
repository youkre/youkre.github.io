---
title: 第3章 N元语言模型
date: '2025-07-19T00:00:00+08:00'
description: ''
draft: false
isCJKLanguage: true
keywords: []
slug: ch3
summary: ''
params:
  virtual: false
hex_id: artl-d7737ef33b71102d
head_level: 1
prev_page:
  hex_id: artl-60933eeabaaa8b01
  slug: ch2-09
  title: 2.9 最小编辑距离（Minimum Edit Distance）
next_page:
  hex_id: artl-bb414c1ca1fe6391
  slug: ch3-01
  title: 3.1 N元模型
---

> “你总是那么迷人！”他微笑着说道，这时我偶尔会鞠个躬，他们也注意到了一辆四匹马拉的马车，心生向往。
>
> 由简·奥斯汀语料的三元模型生成的随机句子

正如一句老话所说：“预测很难——尤其是预测未来。”
但如果尝试预测一些看起来简单得多的事情呢？比如一个人接下来会说什么词。
举个例子，下面这句话接下来可能会出现什么词：

```text
The water of Walden Pond is so beautifully ...
```

你可能会觉得，可能的词是`blue`、`green`或者`clear`，但不太可能是`refrigeratornor`或`this`。
在本章中，我们将引入**语言模型**（Language Models，简称LMs）来精确化这种直觉。
语言模型能够为每一个可能的下一个词分配一个**概率**。
它不仅可以为一个完整的句子分配概率，还能告诉我们，以下这段话在文本中出现的概率：

```text
all of a sudden I notice three guys standing on the sidewalk。
```

比下面这个词语顺序被打乱的版本出现的可能性要大得多：

```text
on guys all I of notice sidewalk three a sudden standing the
```

为什么要去预测下一个词？
主要原因是，大语言模型的构建方式就是通过训练来预测下一个词！！
正如我们将在第 5 到第 10 章看到的那样，仅通过训练大模型通过相邻的词来预测下一个词，就能让模型能够学到丰富的语言知识。

这种概率知识非常重要。
考虑纠正语法或拼写错误，比如在 `Their are two midterms` 中， `There` 被误写成了 `Their`，或者 `Everything has improve` 中，`improve` 应为 `improved`。
因为 `There are` 比 `Their are` 更常见， `has improved` 也比 `has improve` 更有可能出现，所以语言模型可以帮助用户选择更符合语法的表达。

再比如，语音识别系统要判断你说的是 `I will be back soonish`（我很快就回来），而不是 `I will be bassoon dish` （我将成为低音管盘），就需要知道 `back soonish` 这个组合更有可能。
语言模型还可以帮助**增强和替代交流系统**（augmentative and alternative communication，AAC）（Trnka 等，2007；Kane 等，2017）。
一些身体不便、无法说话或打手语的人可以通过注视或其它动作从菜单中选择词语，语言模型可以辅助这类系统预测可能的词。

在本章中，我们将介绍最简单的语言模型：**n元语言模型**（n-gram language model）。
n 元（n-gram）是指由 n 个词组成的序列：比如两个词组成的 2 元（称为**双词模型**或**bigram**），像 `The water` 或 `water of`；三个词组成的3元（**三词模型**或**trigram**），如 `The water of` 或 `water of Walden`。
我们也用“n-gram”这个词来指代一种概率模型(这造成了术语上有一点模糊)，它可以根据前面的 n-1 个词来估计下一个词的概率，从而为整个词序列分配概率。

在后续章节中，我们将介绍更强大的**基于Transformer架构的神经网络大型语言模型**（见第9章）。
但由于 n 元模型的具有非常清晰且易于理解的形式，我们将用它来介绍大型语言模型的一些核心概念，包括**训练集与测试集**、**困惑度**（perplexity）、**采样**（sampling）以及**插值**（interpolation）等。