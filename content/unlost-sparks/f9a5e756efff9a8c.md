---
title: 熵与概率
description: ''
draft: false
isCJKLanguage: true
keywords: []
summary: ''
date: '2025-12-15'
---

## 什么是熵？

在深度学习和自然语言处理中，我们经常会遇到“熵”（entropy）这个概念。那么，熵到底是什么呢？

**熵是信息量的一种度量**。但“信息量”又是什么？

在信息论中，香农（Shannon）提出： 

> 一个事件 <span class="math"><math><mi>x</mi></math></span> 发生时所携带的信息量（也称为“自信息”，self-information）定义为：

<div class="math">
<math display="block"><mi>I</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo lspace="0" rspace="0">−</mo><msub><mo lspace="0" rspace="0.1667em">log</mo><mn>2</mn></msub><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math>
</div>

### 直观理解

- 如果某事几乎必然发生（<span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≈</mo><mn>1</mn></math></span>），那么它发生时**不带来新信息** → <span class="math"><math><mi>I</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≈</mo><mn>0</mn></math></span>；
- 如果某事极不可能发生（<span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>≈</mo><mn>0</mn></math></span>），一旦发生就**非常令人惊讶** → <span class="math"><math><mi>I</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math></span> 很大。

因此：**越意外的事件，信息量越大**。

既然熵是一种可计算的量，那么它的计算公式是什么？在介绍熵的计算之前，我们需要先理解概率论中的**随机变量**及其**期望**。

## 随机变量及其期望

在概率论中，**随机变量**（random variable）并不是一个“普通变量”，而是一个**函数**。具体来说：

- 设有一个**样本空间** <span class="math"><math><mrow><mspace/><mi mathvariant="normal">Ω</mi></mrow></math></span>（即所有可能实验结果的集合）；
- 随机变量 <span class="math"><math><mi>X</mi></math></span> 是一个从 <span class="math"><math><mrow><mspace/><mi mathvariant="normal">Ω</mi></mrow></math></span> 到某个可数集合 <span class="math"><math><mrow><mspace/><mi>𝒳︀</mi></mrow></math></span>（通常是实数集或离散符号集）的函数：

<div class="math">
<math display="block"><mi>X</mi><mo lspace="0.2778em" rspace="0.2778em">:</mo><mrow><mspace/><mi mathvariant="normal">Ω</mi></mrow><mo stretchy="false">→</mo><mrow><mspace/><mi>𝒳︀</mi></mrow></math>
</div>

### 举个例子

抛两枚公平的硬币，用 H 表示正面，T 表示反面，样本空间为：

- <span class="math"><math><mo stretchy="false">(</mo><mi>H</mi><mo>,</mo><mi>H</mi><mo stretchy="false">)</mo></math></span>
- <span class="math"><math><mo stretchy="false">(</mo><mi>H</mi><mo>,</mo><mi>T</mi><mo stretchy="false">)</mo></math></span>
- <span class="math"><math><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>H</mi><mo stretchy="false">)</mo></math></span>
- <span class="math"><math><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>T</mi><mo stretchy="false">)</mo></math></span>

设随机变量 X 表示在一次抛掷两枚硬币的试验中，正面朝上的硬币数量，则：

- <span class="math"><math><mi>X</mi><mo stretchy="false">(</mo><mi>H</mi><mo>,</mo><mi>H</mi><mo stretchy="false">)</mo><mo>=</mo><mn>2</mn></math></span>
- <span class="math"><math><mi>X</mi><mo stretchy="false">(</mo><mi>H</mi><mo>,</mo><mi>T</mi><mo stretchy="false">)</mo><mo>=</mo><mi>X</mi><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>H</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></math></span>
- <span class="math"><math><mi>X</mi><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>T</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></math></span>

于是，<span class="math"><math><mi>X</mi></math></span> 的取值为 <span class="math"><math><mo stretchy="false">{</mo><mn>0</mn><mo>,</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo stretchy="false">}</mo></math></span>

**注意**：当我们说“随机变量 X 表示正面出现的次数”，指的是在一次具体的试验中（如抛两枚硬币得到 (H,T)），X 的取值为该次试验中 H 的数量。这不是指“样本空间中有多少个结果包含 H”——后者是一个固定常数，不是随机变量。这一点对于理解概率论的后续内容非常重要。

我们可以对随机变量的可能取值指定概率。
对于上述抛硬币的例子，我们假设两枚硬币都是公平的（即 <span class="math"><math><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></math></span>），且两次抛掷相互独立，则每个基本结果的概率为各自概率的乘积：

<div class="math">
<math display="block"><mtable displaystyle="true" scriptlevel="0"><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mn>0</mn><mo stretchy="false">)</mo></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>T</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo><mo lspace="0.2222em" rspace="0.2222em">·</mo><mi>P</mi><mo stretchy="false">(</mo><mi>T</mi><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0.2222em" rspace="0.2222em">×</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mfrac><mn>1</mn><mn>4</mn></mfrac></mtd></mtr></mtable></math>
</div>

<div class="math">
<math display="block"><mtable displaystyle="true" scriptlevel="0"><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mn>1</mn><mo stretchy="false">)</mo></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mo stretchy="false">(</mo><mi>H</mi><mo>,</mo><mi>T</mi><mo stretchy="false">)</mo><mo>,</mo><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>H</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>H</mi><mo>,</mo><mi>T</mi><mo stretchy="false">)</mo><mo>+</mo><mi>P</mi><mo stretchy="false">(</mo><mi>T</mi><mo>,</mo><mi>H</mi><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0.2222em" rspace="0.2222em">×</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo>+</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0.2222em" rspace="0.2222em">×</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mfrac><mn>2</mn><mn>4</mn></mfrac></mtd></mtr></mtable></math>
</div>

<div class="math">
<math display="block"><mtable displaystyle="true" scriptlevel="0"><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mn>2</mn><mo stretchy="false">)</mo></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mo stretchy="false">(</mo><mi>H</mi><mo>,</mo><mi>H</mi><mo stretchy="false">)</mo><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mfrac><mn>1</mn><mn>2</mn></mfrac><mo lspace="0.2222em" rspace="0.2222em">×</mo><mfrac><mn>1</mn><mn>2</mn></mfrac></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mfrac><mn>1</mn><mn>4</mn></mfrac></mtd></mtr></mtable></math>
</div>

随机变量的**期望**（expected value）是其所有可能取值的**加权平均**，权重即为对应取值的概率：

<div class="math">
<math display="block"><mi>𝔼</mi><mo stretchy="false">[</mo><mi>X</mi><mo stretchy="false">]</mo><mo>=</mo><munder><mo lspace="0">∑</mo><mrow><mi>x</mi><mo lspace="0" rspace="0">∈</mo><mrow><mspace/><mi>𝒳︀</mi></mrow></mrow></munder><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo lspace="0.2222em" rspace="0.2222em">·</mo><mi>x</mi></math>
</div>

## 熵的计算

> 熵 = 信息量的期望值（即加权平均）

随机变量 <span class="math"><math><mi>X</mi></math></span> 的熵 <span class="math"><math><mi>H</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo></math></span> 定义为其自信息 <span class="math"><math><mi>I</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo></math></span> 的期望：

<div class="math">
<math display="block"><mtable displaystyle="true" scriptlevel="0"><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"><mi>H</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mi>𝔼</mi><mo stretchy="false">[</mo><mi>I</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo><mo stretchy="false">]</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><munder><mo lspace="0">∑</mo><mrow><mi>x</mi><mo lspace="0" rspace="0">∈</mo><mrow><mspace/><mi>𝒳︀</mi></mrow></mrow></munder><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo lspace="0.2222em" rspace="0.2222em">·</mo><mi>I</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><munder><mo lspace="0">∑</mo><mrow><mi>x</mi><mo lspace="0" rspace="0">∈</mo><mrow><mspace/><mi>𝒳︀</mi></mrow></mrow></munder><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo lspace="0.2222em" rspace="0.2222em">·</mo><mo minsize="1.2em" maxsize="1.2em">(</mo><mo>−</mo><msub><mo lspace="0" rspace="0.1667em">log</mo><mn>2</mn></msub><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo minsize="1.2em" maxsize="1.2em">)</mo></mtd></mtr><mtr><mtd style="text-align: right;justify-items: end;padding-right: 0;"></mtd><mtd style="text-align: left;justify-items: start;padding-left: 0;"><mo>=</mo><mo lspace="0" rspace="0">−</mo><munder><mo lspace="0">∑</mo><mrow><mi>x</mi><mo lspace="0" rspace="0">∈</mo><mrow><mspace/><mi>𝒳︀</mi></mrow></mrow></munder><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><msub><mo lspace="0.1667em" rspace="0.1667em">log</mo><mn>2</mn></msub><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></mtd></mtr></mtable></math>
</div>

换句话说，**熵是以概率 <span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math></span> 为权重，对每个结果的信息量 <span class="math"><math><mo lspace="0" rspace="0">−</mo><msub><mo lspace="0" rspace="0.1667em">log</mo><mn>2</mn></msub><mi>p</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math></span> 所做的加权平均**。

这与算术平均 <span class="math"><math><mfrac><mn>1</mn><mi>n</mi></mfrac><mo>∑</mo><msub><mi>x</mi><mi>i</mi></msub></math></span> 不同——**高频事件对熵的贡献更大**。

### 举个例子：抛硬币

#### 情况一：公平硬币

- <span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mtext>H</mtext><mo stretchy="false">)</mo><mo>=</mo><mi>p</mi><mo stretchy="false">(</mo><mtext>T</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn></math></span>
- 信息量：<span class="math"><math><mi>I</mi><mo stretchy="false">(</mo><mtext>H</mtext><mo stretchy="false">)</mo><mo>=</mo><mi>I</mi><mo stretchy="false">(</mo><mtext>T</mtext><mo stretchy="false">)</mo><mo>=</mo><mo lspace="0" rspace="0">−</mo><msub><mo lspace="0" rspace="0.1667em">log</mo><mn>2</mn></msub><mn>0.5</mn><mo>=</mo><mn>1</mn></math></span> 比特
- 熵：<span class="math"><math><mi>H</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0.5</mn><mo lspace="0.2222em" rspace="0.2222em">×</mo><mn>1</mn><mo>+</mo><mn>0.5</mn><mo lspace="0.2222em" rspace="0.2222em">×</mo><mn>1</mn><mo>=</mo><mn>1</mn></math></span> 比特

平均每次抛硬币带来 **1 比特**信息。

#### 情况二：偏置硬币

- <span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mtext>H</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.9</mn><mo>,</mo><mspace width="0.2777778em"/><mi>p</mi><mo stretchy="false">(</mo><mtext>T</mtext><mo stretchy="false">)</mo><mo>=</mo><mn>0.1</mn></math></span>
- <span class="math"><math><mi>I</mi><mo stretchy="false">(</mo><mtext>H</mtext><mo stretchy="false">)</mo><mo>=</mo><mo lspace="0" rspace="0">−</mo><msub><mo lspace="0" rspace="0.1667em">log</mo><mn>2</mn></msub><mn>0.9</mn><mo>≈</mo><mn>0.15</mn></math></span> 比特（不意外）
- <span class="math"><math><mi>I</mi><mo stretchy="false">(</mo><mtext>T</mtext><mo stretchy="false">)</mo><mo>=</mo><mo lspace="0" rspace="0">−</mo><msub><mo lspace="0" rspace="0.1667em">log</mo><mn>2</mn></msub><mn>0.1</mn><mo>≈</mo><mn>3.32</mn></math></span> 比特（很意外）
- 熵：<span class="math"><math><mi>H</mi><mo stretchy="false">(</mo><mi>X</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0.9</mn><mo lspace="0.2222em" rspace="0.2222em">×</mo><mn>0.15</mn><mo>+</mo><mn>0.1</mn><mo lspace="0.2222em" rspace="0.2222em">×</mo><mn>3.32</mn><mo>≈</mo><mn>0.47</mn></math></span> 比特

尽管“出现 T”的信息量很大，但由于它极少发生，**平均信息量反而更低**。

这正是“加权平均”的体现：**高频事件主导了熵的大小**。

## 小结

- 熵不是单个事件的信息量，而是**整个概率分布的平均不确定性**；
- 它衡量的是：**如果你根据这个分布不断抽样，平均每次能获得多少“新信息”**；
- 在语言模型中，熵越低，说明模型对下一个词越“确定”——这正是困惑度（perplexity）的理论基础。