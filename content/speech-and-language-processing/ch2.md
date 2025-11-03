---
title: "第2章 正则表达式、词元化与编辑距离"
summary: ""
date: 2025-10-14T23:46:00+08:00
---

> User: I am unhappy.
>
> ELIZA: DO YOU THINK COMING HERE WILL HELP YOU NOT TO BE UNHAPPY
>
> User: I need some help, that much seems certain.
>
> ELIZA: WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP
>
> User: Perhaps I could learn to get along with my mother.
>
> ELIZA: TELL ME MORE ABOUT YOUR FAMILY
>
> User: My mother takes care of me.
>
> ELIZA: WHO ELSE IN YOU FAMILY TAKES CARE OF YOU
>
> User: My father.
>
> ELIZA: YOUR FATHER
>
> User: You are like my father in some ways.
>
> Weizenbaum (1966)

上述对话来自 **ELIZA**，这是一个早期的自然语言处理系统，能够通过模仿罗杰式心理治疗师的回应方式与用户进行有限的对话（Weizenbaum, 1966）。ELIZA 是一个出人意料地简单的程序，它使用模式匹配来识别诸如“I need X”之类的短语，并将其转换为“WHAT WOULD IT MEAN TO YOU IF YOU GOT SOME HELP”这样的适当输出。这种简单技术在该领域之所以成功，是因为 ELIZA 实际上并不需要真正了解任何东西就能模仿罗杰式心理治疗师。正如 Weizenbaum 所指出的，这是少数几种倾听者可以表现得对世界一无所知的对话类型之一。ELIZA 对人类对话的模仿取得了惊人的成功：许多与之互动的人开始相信它真的理解了他们及其问题；即使在程序的工作原理被解释之后，许多人仍继续相信 ELIZA 的能力（Weizenbaum, 1976）；时至今日，这类**聊天机器人**（chatbots）仍然是一种有趣的消遣。

当然，现代对话系统远不止是一种消遣；它们能够回答问题、预订航班或查找餐厅，为此它们对用户意图有更复杂的理解，我们将在第15章中看到这一点。尽管如此，驱动 ELIZA 和其他聊天机器人基于模式的简单方法，在自然语言处理中仍扮演着至关重要的角色。

我们将从描述文本模式最重要的工具开始：**正则表达式**（regular expression）。正则表达式可用于指定我们可能想要从文档中提取的字符串，其应用范围广泛，从上面 ELIZA 中将“I need X”转换为某种形式，到定义如 $199 或 $24.99 这样的字符串以从文档中提取价格表。

接下来，我们将转向一组统称为**文本归一化**（text normalization）的任务，其中正则表达式起着重要作用。文本归一化是指将文本转换为更方便、更标准的形式。例如，我们对语言的大多数处理都依赖于首先将词语从连续文本中分离出来，即**词元化**（tokenizing）任务。英语单词通常由空格分隔，但空格并不总是足够。像“New York”和“rock ’n’ roll”这样的词有时被视为一个整体词，尽管它们包含空格；而有时我们又需要将 *I’m* 拆分为 I 和 am 两个词。在处理推文或短信时，我们需要对 `:)` 这样的**表情符号**（emoticons）或 `#nlproc` 这样的**话题标签**（hashtags）进行词元化。有些语言，如日语，词与词之间没有空格，因此词元化变得更加困难。

文本归一化的另一个部分是**词形还原**（lemmatization），即判断两个词尽管表面形式不同，但具有相同词根的任务。例如，sang、sung 和 sings 都是动词 sing 的不同形式。sing 是这些词的共同*词元*（lemma），而一个**词形还原器**（lemmatizer）会将所有这些形式映射回 sing。词形还原对于处理阿拉伯语等形态复杂的语言至关重要。**词干提取**（Stemming）指的是词形还原本的一种简化版本，其中我们主要只是去掉词尾的后缀。文本归一化还包括**句子切分**（sentence segmentation）：利用句号或感叹号等线索将文本分割成单个句子。

最后，我们需要比较词语和其他字符串。我们将介绍一种称为**编辑距离**（edit distance）的度量方法，它基于将一个字符串转换为另一个字符串所需的编辑操作（插入、删除、替换）次数来衡量两个字符串的相似程度。编辑距离是一种在语言处理中广泛应用的算法，从拼写纠正、语音识别到共指消解都有其身影。


<nav class="pagination justify-content-between">
<span></span>
<a href="../">目录</a>
<a href="../ch2-02">2.2 词</a>
</nav>

