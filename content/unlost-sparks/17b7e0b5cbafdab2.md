---
title: 累积分布函数（CDF）完整参考
description: 一份自包含的 CDF 参考页
draft: false
isCJKLanguage: true
keywords: []
summary: 一份自包含的 CDF 参考页
date: '2026-07-21'
---

《概率论导论》一书中虽然语言浅显，但是很多概念定义不清晰，例如：

**定义 3.6.1** 随机变量 X 的累积分布函数是一个函数 F_X(x)，其中 <span class="math"><math><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>≤</mo><mi>x</mi><mo stretchy="false">)</mo></math></span>。当不存在模糊概念的风险时，有时候可以去掉下标，只用 F（获其他字母）表示一个累积分布函数。

这里的问题是，定义只有两句话，然后就什么都没有了，这两句话里还有一半根本不应该放在定义里。后面就是例子，例子只是在介绍怎么从 CDF 转换到 PMF，但是例子不足以当作一个通用的东西来使用，以及没有给出足够多的说明来强调其特点。

这种重新完善一下这部分内容。

## 1. 定义

对于任意随机变量 <span class="math"><math><mi>X</mi></math></span>，其累积分布函数（CDF）定义为：

<div class="math">
<math display="block"><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>≤</mo><mi>x</mi><mo stretchy="false">)</mo><mo>,</mo><mspace width="1em"/><mi>x</mi><mo>∈</mo><mi>ℝ</mi></math>
</div>

- **输入**：任意实数 <span class="math"><math><mi>x</mi></math></span>
- **输出**：概率 <span class="math"><math><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>≤</mo><mi>x</mi><mo stretchy="false">)</mo></math></span>，是一个介于 0 和 1 之间的数
- **下标说明**：当不会混淆时，可简写为 <span class="math"><math><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math></span>

> 注意：CDF 对所有实数 <span class="math"><math><mi>x</mi></math></span> 都有定义，不限于支撑集上的点。

## 2. 离散型 CDF 与 PMF 的关系

若 <span class="math"><math><mi>X</mi></math></span> 是离散型随机变量，支撑为 <span class="math"><math><mo stretchy="false">{</mo><msub><mi>a</mi><mn>1</mn></msub><mo>,</mo><msub><mi>a</mi><mn>2</mn></msub><mo>,</mo><mo lspace="0">.</mo><mo lspace="0" rspace="0.1667em">.</mo><mo lspace="0" rspace="0">.</mo><mo stretchy="false">}</mo></math></span>，PMF 为 <span class="math"><math><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo><mo>=</mo><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mi>k</mi><mo stretchy="false">)</mo></math></span>，则：

**正向：从 PMF 到 CDF**

<div class="math">
<math display="block"><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><munder><mo lspace="0">∑</mo><mrow><mi>k</mi><mo lspace="0" rspace="0">:</mo><mi>k</mi><mo lspace="0" rspace="0">≤</mo><mi>x</mi></mrow></munder><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo></math>
</div>
即把所有不超过 <span class="math"><math><mi>x</mi></math></span> 的取值对应的概率相加。

**反向：从 CDF 到 PMF**

<div class="math">
<math display="block"><msub><mi>p</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo><mo>=</mo><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo><mo>−</mo><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><msup><mi>k</mi><mo>−</mo></msup><mo stretchy="false">)</mo></math>
</div>

其中 <span class="math"><math><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><msup><mi>k</mi><mo>−</mo></msup><mo stretchy="false">)</mo><mo>=</mo><munder><mo movablelimits="true" lspace="0" rspace="0.1667em">lim</mo><mrow><mi>t</mi><mo stretchy="false" lspace="0" rspace="0">→</mo><msup><mi>k</mi><mo>−</mo></msup></mrow></munder><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo></math></span>，即 <span class="math"><math><msub><mi>F</mi><mi>X</mi></msub></math></span> 在 <span class="math"><math><mi>k</mi></math></span> 处的左极限。

这个差值是 CDF 在点 <span class="math"><math><mi>k</mi></math></span> 处的**跳跃高度**，也就是落在该点上的概率质量。

## 3. CDF 的三个基本性质

对于任何随机变量（离散或连续）的 CDF <span class="math"><math><mi>F</mi></math></span>：

| 性质 | 数学表达 | 含义 |
|---|---|---|
| **非降** | <span class="math"><math><msub><mi>x</mi><mn>1</mn></msub><mo>&lt;</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">⇒</mo><mi>F</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mn>1</mn></msub><mo stretchy="false">)</mo><mo>≤</mo><mi>F</mi><mo stretchy="false">(</mo><msub><mi>x</mi><mn>2</mn></msub><mo stretchy="false">)</mo></math></span> | 概率不会随着阈值增大而减少 |
| **右连续** | <span class="math"><math><munder><mo movablelimits="true" lspace="0" rspace="0.1667em">lim</mo><mrow><mi>t</mi><mo stretchy="false" lspace="0" rspace="0">→</mo><msup><mi>x</mi><mo>+</mo></msup></mrow></munder><mi>F</mi><mo stretchy="false">(</mo><mi>t</mi><mo stretchy="false">)</mo><mo>=</mo><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo></math></span> | 在任意点处，右侧极限等于函数值 |
| **极限** | <span class="math"><math><munder><mo movablelimits="true" lspace="0" rspace="0.1667em">lim</mo><mrow><mi>x</mi><mo stretchy="false" lspace="0" rspace="0">→</mo><mo lspace="0" rspace="0">−</mo><mi>∞</mi></mrow></munder><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></math></span>，<span class="math"><math><munder><mo movablelimits="true" lspace="0" rspace="0.1667em">lim</mo><mrow><mi>x</mi><mo stretchy="false" lspace="0" rspace="0">→</mo><mo lspace="0" rspace="0">+</mo><mi>∞</mi></mrow></munder><mi>F</mi><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></math></span> | 概率从 0 积累到 1 |

> 这三个性质是判断一个函数是否是合法的 CDF 的依据。

## 4. 离散型 CDF 的计算步骤模板

当你需要计算一个离散型随机变量的 CDF 时，按以下步骤操作：

### 步骤 1：明确支撑集
写出 <span class="math"><math><mi>X</mi></math></span> 的所有可能取值集合 <span class="math"><math><msub><mi>S</mi><mi>X</mi></msub></math></span>。

### 步骤 2：写出 PMF
如果没有直接给出，先确定 <span class="math"><math><mi>P</mi><mo stretchy="false">(</mo><mi>X</mi><mo>=</mo><mi>k</mi><mo stretchy="false">)</mo></math></span> 对每个 <span class="math"><math><mi>k</mi><mo>∈</mo><msub><mi>S</mi><mi>X</mi></msub></math></span> 的表达式。

### 步骤 3：确定阈值 <span class="math"><math><mi>x</mi></math></span> 的位置
判断 <span class="math"><math><mi>x</mi></math></span> 相对于支撑集的位置：

- 若 <span class="math"><math><mi>x</mi><mo>&lt;</mo><mo lspace="0" rspace="0">min</mo><mo stretchy="false">(</mo><msub><mi>S</mi><mi>X</mi></msub><mo stretchy="false">)</mo></math></span>：<span class="math"><math><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></math></span>
- 若 <span class="math"><math><mi>x</mi><mo>≥</mo><mo lspace="0" rspace="0">max</mo><mo stretchy="false">(</mo><msub><mi>S</mi><mi>X</mi></msub><mo stretchy="false">)</mo></math></span>：<span class="math"><math><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></math></span>
- 若 <span class="math"><math><mi>x</mi></math></span> 落在支撑集内部：找出所有满足 <span class="math"><math><mi>k</mi><mo>≤</mo><mi>x</mi></math></span> 的 <span class="math"><math><mi>k</mi></math></span>，累加其概率

### 步骤 4：写成带 <span class="math"><math><mo stretchy="false">⌊</mo><mi>x</mi><mo stretchy="false">⌋</mo></math></span> 或分段的形式
通常结果是一个关于 <span class="math"><math><mi>x</mi></math></span> 的分段函数。

## 5. 常见离散分布的 CDF

| 分布 | PMF | CDF |
|---|---|---|
| **离散均匀** <span class="math"><math><mo stretchy="false">{</mo><mn>1</mn><mo>,</mo><mo lspace="0">.</mo><mo lspace="0" rspace="0.1667em">.</mo><mo lspace="0">.</mo><mo>,</mo><mi>n</mi><mo stretchy="false">}</mo></math></span> | <span class="math"><math><mn>1</mn><mo lspace="0" rspace="0">/</mo><mi>n</mi></math></span> | <span class="math"><math><mn>0</mn></math></span>（<span class="math"><math><mi>x</mi><mo>&lt;</mo><mn>1</mn></math></span>），<span class="math"><math><mo stretchy="false">⌊</mo><mi>x</mi><mo stretchy="false">⌋</mo><mo lspace="0" rspace="0">/</mo><mi>n</mi></math></span>（<span class="math"><math><mn>1</mn><mo>≤</mo><mi>x</mi><mo>&lt;</mo><mi>n</mi></math></span>），<span class="math"><math><mn>1</mn></math></span>（<span class="math"><math><mi>x</mi><mo>≥</mo><mi>n</mi></math></span>） |
| **伯努利** <span class="math"><math><mtext>Bern</mtext><mo stretchy="false">(</mo><mi>p</mi><mo stretchy="false">)</mo></math></span> | <span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mn>0</mn><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mo>−</mo><mi>p</mi></math></span>，<span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mn>1</mn><mo stretchy="false">)</mo><mo>=</mo><mi>p</mi></math></span> | <span class="math"><math><mn>0</mn></math></span>（<span class="math"><math><mi>x</mi><mo>&lt;</mo><mn>0</mn></math></span>），<span class="math"><math><mn>1</mn><mo>−</mo><mi>p</mi></math></span>（<span class="math"><math><mn>0</mn><mo>≤</mo><mi>x</mi><mo>&lt;</mo><mn>1</mn></math></span>），<span class="math"><math><mn>1</mn></math></span>（<span class="math"><math><mi>x</mi><mo>≥</mo><mn>1</mn></math></span>） |
| **泊松** <span class="math"><math><mtext>Pois</mtext><mo stretchy="false">(</mo><mi>λ</mi><mo stretchy="false">)</mo></math></span> | <span class="math"><math><msup><mi>e</mi><mrow><mo lspace="0" rspace="0">−</mo><mi>λ</mi></mrow></msup><msup><mi>λ</mi><mi>k</mi></msup><mo lspace="0" rspace="0">/</mo><mi>k</mi><mo>!</mo></math></span> | <span class="math"><math><msup><mi>e</mi><mrow><mo lspace="0" rspace="0">−</mo><mi>λ</mi></mrow></msup><munderover><mo>∑</mo><mrow><mi>k</mi><mo lspace="0" rspace="0">=</mo><mn>0</mn></mrow><mrow><mo stretchy="false">⌊</mo><mi>x</mi><mo stretchy="false">⌋</mo></mrow></munderover><msup><mi>λ</mi><mi>k</mi></msup><mo lspace="0" rspace="0">/</mo><mi>k</mi><mo>!</mo></math></span>（<span class="math"><math><mi>x</mi><mo>≥</mo><mn>0</mn></math></span>） |

## 6. 一个完整的例题流程

**题目**：<span class="math"><math><mi>X</mi><mo>∼</mo><mtext>DUni</mtext><mo stretchy="false">(</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mo lspace="0">.</mo><mo lspace="0" rspace="0.1667em">.</mo><mo lspace="0">.</mo><mo>,</mo><mi>n</mi><mo stretchy="false">)</mo></math></span>，求 CDF。

1. 支撑集：<span class="math"><math><msub><mi>S</mi><mi>X</mi></msub><mo>=</mo><mo stretchy="false">{</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mo lspace="0">.</mo><mo lspace="0" rspace="0.1667em">.</mo><mo lspace="0">.</mo><mo>,</mo><mi>n</mi><mo stretchy="false">}</mo></math></span>，PMF：<span class="math"><math><mi>p</mi><mo stretchy="false">(</mo><mi>k</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn><mo lspace="0" rspace="0">/</mo><mi>n</mi></math></span>
2. 若 <span class="math"><math><mi>x</mi><mo>&lt;</mo><mn>1</mn></math></span>：无 <span class="math"><math><mi>k</mi><mo>≤</mo><mi>x</mi></math></span> → <span class="math"><math><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>0</mn></math></span>
3. 若 <span class="math"><math><mn>1</mn><mo>≤</mo><mi>x</mi><mo>&lt;</mo><mi>n</mi></math></span>：<span class="math"><math><mi>k</mi><mo>=</mo><mn>1</mn><mo>,</mo><mn>2</mn><mo>,</mo><mo lspace="0">.</mo><mo lspace="0" rspace="0.1667em">.</mo><mo lspace="0">.</mo><mo>,</mo><mo stretchy="false">⌊</mo><mi>x</mi><mo stretchy="false">⌋</mo></math></span> → <span class="math"><math><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mo stretchy="false">⌊</mo><mi>x</mi><mo stretchy="false">⌋</mo><mo lspace="0" rspace="0">/</mo><mi>n</mi></math></span>
4. 若 <span class="math"><math><mi>x</mi><mo>≥</mo><mi>n</mi></math></span>：所有 <span class="math"><math><mi>k</mi></math></span> 都满足 → <span class="math"><math><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mn>1</mn></math></span>

最终：

<div class="math">
<math display="block"><msub><mi>F</mi><mi>X</mi></msub><mo stretchy="false">(</mo><mi>x</mi><mo stretchy="false">)</mo><mo>=</mo><mrow><mo>{</mo><mtable><mtr><mtd style="text-align: left;justify-items: start;padding-right: 0;"><mn>0</mn><mo rspace="0">,</mo></mtd><mtd style="text-align: left;justify-items: start;padding-right: 0;padding-left:1em;"><mi>x</mi><mo>&lt;</mo><mn>1</mn></mtd></mtr><mtr><mtd style="text-align: left;justify-items: start;padding-right: 0;"><mfrac displaystyle="true"><mrow><mo stretchy="false">⌊</mo><mi>x</mi><mo stretchy="false">⌋</mo></mrow><mi>n</mi></mfrac><mo rspace="0">,</mo></mtd><mtd style="text-align: left;justify-items: start;padding-right: 0;padding-left:1em;"><mn>1</mn><mo>≤</mo><mi>x</mi><mo>&lt;</mo><mi>n</mi></mtd></mtr><mtr><mtd style="text-align: left;justify-items: start;padding-right: 0;"><mn>1</mn><mo rspace="0">,</mo></mtd><mtd style="text-align: left;justify-items: start;padding-right: 0;padding-left:1em;"><mi>x</mi><mo>≥</mo><mi>n</mi></mtd></mtr></mtable><mo>⁣</mo></mrow></math>
</div>