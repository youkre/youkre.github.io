---
title: "第4章 罗辑回归与文本分类"
summary: ""
date: 2025-12-22T14:30:00+08:00
---

> En sus remotas p´aginas est´a escrito que los animales se dividen en:  
> a. pertenecientes al Emperador  
> b. embalsamados  
> c. amaestrados  
> d. lechones  
> e. sirenas  
> f. fabulosos  
> g. perros sueltos  
> h. incluidos en esta clasiﬁcaci ´on  
> i. que se agitan como locos.  
> j. innumerables.  
> k. dibujados con un pincel ﬁn ´ısimo de pelo de camello  
> l. etc ´etera.  
> m. que acaban de romper el jarr ´on  
> n. que de lejos parecen moscas
>
> Borges (1964)

**分类**（Classification），即是人类智能的核心，也是机器智能的核心。
判断感官接收到的是哪个字母、单词或图像，识别面孔或声音，分拣邮件，给作业评分——这些例子都是为输入信息分类。
作家豪尔赫·路易斯·博尔赫斯（Jorge Luis Borges, 1964）曾以寓言的方式突显了这一任务的潜在挑战，他想象将动物分为以下类别：

* (a) 属于皇帝的动物，
* (b) 经过防腐处理的动物，
* (c) 受过训练的动物，
* (d) 吃奶的小猪，
* (e) 美人鱼，
* (f) 传说中的动物，
* (g) 流浪狗，
* (h) 被包含在此分类中的动物，
* (i) 像疯了一样发抖的动物，
* (j) 数不清的动物，
* (k) 用极细的骆驼毛笔画出的动物，
* (l) 其他动物，
* (m) 刚刚打碎花瓶的动物，
* (n) 从远处看像苍蝇的动物。

幸运的是，我们在语言处理中使用的的分类标准远比博尔赫斯的分类容易定义。
在本章中，我们将介绍用于分类的**罗辑回归**（logistic regression）算法，并将其应用于**文本分类**（text categorization），即为整篇文本或文档分配一个标签（label）或类别(category)。
我们重点关注一种文本分类任务：**情感分析**（sentiment analysis），即对情感分类，这些情感是作者对某个对象所表达正面或负面的态度。
电影、书籍或产品的评论表达了作者对这些产品的看法，而社论或政治性文本则表达了对某项政治行动或某位候选人的情感倾向。
因此，从市场营销到政治等众多领域，提取情感都具有重要意义。

在将文本标注为表达正面或负面立场的二分类任务中，某些词语（如 *awesome* 和 *love*，或 *awful* 和 *ridiculously*）具有很强的判别性，这一点从以下电影/餐厅评论的示例片段中可见一斑：

> \+ ...awesome caramel sauce and sweet toasty almonds. I love this place!  
> − ...awful pizza and ridiculously overpriced...

文本分类任务有很多中。
在**垃圾邮件检测**（Spam detection）中，我们将电子邮件归类为“垃圾邮件”或“非垃圾邮件”。
**语言识别**（language id）任务确定某种文本使用什么语言书写的，而**作者归属**（authorship attribution）用于确定文本的作者，在人文和司法分析中都具有重要价值。

但分类之所以如此重要，还在于**语言建模**（language modeling）本身也可以被看作一种分类任务：每个词都可以被视为一个类别，因此预测下一个词，本质上就是将当前已有的上下文（context-so-far）分类到各个可能的“下一个词”类别中。
正如我们后续将看到的，这一观点正是大型语言模型（large language models）的核心原理。

本章所介绍的分类算法是逻辑回归（logistic regression），同样具有多方面的关键意义。  
首先，逻辑回归与神经网络关系密切。
正如第6章将要展示的，神经网络可以被理解为多个逻辑回归分类器层层堆叠而成。  
其次，逻辑回归引入了若干对神经网络和语言模型至关重要的基本概念，例如 **sigmoid 函数**、**softmax 函数**、**logit**，以及用于模型学习的核心算法——**梯度下降**（gradient descent）。  
最后，逻辑回归本身也是社会科学和自然科学中最重要的分析工具之一。


<nav class="pagination justify-content-between">
<a href="../ch3-07">3.7 进阶：困惑度与熵的关系</a>
<a href="../">目录</a>
<a href="../ch4-01">4.1 机器学习与分类</a>
</nav>

