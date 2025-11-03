---
title: "注意力机制：让神经网络学会“重点回顾”"
summary: "让解码器在生成每个词时，能“回头看”输入序列，自动找到最相关的部分，它解决了 Seq2Seq 的核心痛点，上下文向量容量有限，长句子信息丢失严重"
date: 2025-10-08T21:59:00+08:00
---

> 上一篇，我们教会了模型做“中译英”——它能把一句话读懂，写成一张“小纸条”，再翻译出来。  
> 但有个问题：如果句子很长，比如《出师表》全文，这张“小纸条”能记得住所有细节吗？  
> 显然不能。  
> 结果就是：翻译到后面，它忘了前面说了啥……  
>
> 今天，我们就来给它升级技能，让它学会**“重点回顾”**——这就是 **注意力机制（Attention）**。

## 一句话理解注意力机制

> **注意力机制，就是让解码器在生成每个词时，能“回头看”输入序列，自动找到最相关的部分。**  
> 就像你写作文时，不是凭空瞎编，而是不断翻看参考资料，重点摘录关键句子。

它解决了 Seq2Seq 的核心痛点：**“小纸条”（上下文向量）容量有限，长句子信息丢失严重**。

## 从“小纸条”到“重点回顾”：注意力的诞生

传统的 Seq2Seq 模型像这样工作：

```
输入：["I", "love", "deep", "learning"]
                ↓
        编码器 → 压缩 → [h₁, h₂, h₃, h₄] → c（一个向量）
                ↓
        解码器 ← c ← 生成 ["J'aime", "le", "deep", "learning"]
```

问题来了：

- `c` 是一个固定长度的向量，无论输入多长，它都一样大。  
- 输入越长，信息被压缩得越狠，细节就越容易丢失。

**注意力机制的智慧在于：它不再依赖单一的“小纸条”**。

而是让解码器在每一步都做这件事：

> “我现在要生成‘J'aime’了，我应该重点关注输入里的哪些词？”

它会计算一个**权重分布**（soft alignment），比如：

| 输入词 | I | love | deep | learning |
| :--- | :-: | :--: | :--: | :------: |
| 权重 | 0.1 | 0.4 | 0.3 | 0.2 |

然后，用这些权重对编码器的所有隐藏状态加权求和，得到一个**专属的上下文向量 `c_t`**：

$$
c_t = 0.1 \cdot h_1 + 0.4 \cdot h_2 + 0.3 \cdot h_3 + 0.2 \cdot h_4
$$

这个 `c_t` 就是“当前最该关注的信息”。解码器用它来生成下一个词。

**所以，注意力的本质是：动态地、有选择地关注输入信息。**

## 注意力的两种“打分”方式

怎么衡量“当前状态”和“每个输入词”的相关性？常用两种方法：

### 1. 点积注意力（Dot-Product Attention）

最简单粗暴：

$$
\text{score} = s_t \cdot h_i
$$

- `s_t`：解码器当前隐藏状态（“我现在想生成什么”）
- `h_i`：编码器第 i 个隐藏状态（“输入第 i 个词说了什么”）
- 直接向量点积，分数越高越相关。

**优点**：快，适合 GPU 并行。  
**缺点**：要求 `s_t` 和 `h_i` 在同一空间。

### 2. 加性注意力（Additive Attention）——我们实现的版本

更灵活，更强大，是早期注意力的经典设计。

它用一个**小型神经网络**来“学习”相关性：

$$
e_{t,i} = v^T \tanh(W [h_i; s_t])
$$

拆解来看：

- `[h_i; s_t]`：把输入词状态和当前目标状态**拼接**起来。
- `W[...]`：线性变换，映射到新空间。
- `tanh`：非线性激活，增加表达能力。
- `v^T`：打分向量，把高维表示压缩成一个**相似度分数**。

这就像一个“对齐模型”，专门学习“中文词”和“英文词”之间的对应关系。

## PyTorch 实现：手写注意力层

我们来手动实现一个加性注意力模块，这比直接调用 `nn.MultiheadAttention` 更能理解原理。

```python
class AdditiveAttention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.W = nn.Linear(hidden_size, hidden_size)  # 变换矩阵
        self.v = nn.Linear(hidden_size, 1)            # 打分向量

    def forward(self, enc_outputs: torch.Tensor, dec_hidden_states: torch.Tensor):
        """
        enc_outputs: (B, T_enc, H)  编码器所有隐藏状态
        dec_hidden_states: (B, T_dec, H)  解码器所有隐藏状态
        return: context_vectors (B, T_dec, H), attn_weights (B, T_dec, T_enc)
        """
        # Step 1: 变换编码器和解码器状态
        W_enc = self.W(enc_outputs)     # (B, T_enc, H)
        W_dec = self.W(dec_hidden_states) # (B, T_dec, H)

        # Step 2: 扩展维度，准备广播相加
        W_enc_exp = W_enc.unsqueeze(1)  # (B, 1, T_enc, H)
        W_dec_exp = W_dec.unsqueeze(2)  # (B, T_dec, 1, H)

        # Step 3: 广播相加 + tanh → (B, T_dec, T_enc, H)
        energy = torch.tanh(W_enc_exp + W_dec_exp)

        # Step 4: 打分 → (B, T_dec, T_enc, 1)
        scores = self.v(energy)

        # Step 5: 去掉最后一维 → (B, T_dec, T_enc)
        scores = scores.squeeze(-1)

        # Step 6: softmax 归一化，得到注意力权重
        attn_weights = torch.softmax(scores, dim=2)  # (B, T_dec, T_enc)

        # Step 7: 加权求和，得到上下文向量
        context_vectors = torch.bmm(attn_weights, enc_outputs)  # (B, T_dec, H)

        return context_vectors, attn_weights
```

> **关键点**：`torch.bmm` 是 batch matrix multiplication，对每个样本独立做矩阵乘法。

## 集成注意力：升级你的 Seq2Seq 模型

编码器其实没有变化：

```python
class AttentionEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)

    def forward(self, xs: torch.Tensor, input_lengths: Optional[torch.Tensor] = None):
        """
        xs: (batch_size, seq_len)
        Returns: 
            output: (batch_size, seq_len, hidden_size)
            (hn, cn): ((1, batch_size, hidden_size), (1, batch_size, hidden_size))
        """
        # (batch_size, seq_len, embedding_dim)
        embedded = self.embedding(xs)

        if input_lengths is not None:
            packed_embeded = pack_padded_sequence(
                embedded, 
                input_lengths, 
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, (hn, cn) = self.lstm(packed_embeded)
            output, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            # output: (batch_size, seq_len, hidden_size)
            # hn: (1, batch_size, hidden_size)
            # cn: (1, batch_size, hidden_size)
            output, (hn, cn) = self.lstm(embedded)

        return output, (hn, cn)
```

把注意力模块集成到解码器中：

```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.attention = AdditiveAttention(hidden_size)
        self.affine = nn.Linear(2 * hidden_size, vocab_size)  # 拼接 context + hidden

    def forward(self, xs: torch.Tensor, enc_outputs: torch.Tensor, h_c: Tuple[torch.Tensor, torch.Tensor]):
        """
        xs: (B, T)
        enc_outputs: (B, T_enc, H)
        h_c: tuple of (h_0, c_0)
            h_0: (1,  B, H)
            c_0: (1, B, H)
        Returns:
            logits: (B, T, V)
            (h_n, c_n): 最终的隐藏状态
        """
        # xs: (batch_size, seq_len, hidden_size)
        xs = self.embedding(xs)

        if input_lengths is not None:
            packed_embeded = pack_padded_sequence(
                xs, 
                input_lengths, 
                batch_first=True,
                enforce_sorted=False
            )
            packed_output, (hn, cn) = self.lstm(packed_embeded, h_c)
            xs, _ = pad_packed_sequence(packed_output, batch_first=True)
        else:
            # xs: (batch_size, seq_len, hidden_size)
            # hn: (1, batch_size, hidden_size)
            # cn: (1, batch_size, hidden_size)
            xs, (hn, cn) = self.lstm(xs, h_c)

        # 计算注意力
        context_vectors, attn_weights = self.attention(enc_outputs, output)  # (B, T_dec, H)

        # 拼接上下文向量和 LSTM 输出
        out = torch.cat([context_vectors, output], dim=-1)  # (B, T_dec, 2H)

        logits = self.affine(out)
        return logits, (hn, cn), attn_weights  # 返回注意力权重，可用于可视化

    def generate(
            self,
            enc_outputs: torch.Tensor,
            h_c: Tuple[torch.Tensor, torch.Tensor], 
            start_id: int, 
            sample_size: int,
            end_id: Optional[int] = None):
        """
        生成文本（使用注意力）
        enc_outputs: (1, T_enc, H) 编码器输出（batch_size=1）
        h_c: 初始隐藏状态 (h_0, c_0) 初始状态 (1, 1, H)
        start_id: 起始 token ID
        sample_size: 生成多少个词
        end_id: 结束 token ID（可选）
        """
        sampled: List[int] = []
        x = torch.tensor([[start_id]]) # (1, 1)
        h, c = h_c
        sample_id = start_id

        for _ in range(sample_size):
            if end_id is not None and sample_id == end_id:
                break

            out = self.embedding(x) # (1, 1, D)
            out, (h, c) = self.lstm(out, (h, c)) # 更新 h, c (1, 1, H)

            # 关键：使用当前 decoder hidden state 查询 encoder outputs
            context, _ = self.attention(enc_outputs, out)
            combined = torch.cat([context, out], dim=-1) # (1, 1, 2H)

            # Predict
            logits = self.affine(combined) # (1, 1, V)
            sample_id = logits.argmax(dim=-1).item() # 取最大概率的词
            sampled.append(int(sample_id))
            x = torch.tensor([[sample_id]]) # 用于下一次输入

        return sampled
```

注意：

- 输入是 `enc_outputs`（编码器所有隐藏状态），不再是单一的 `hn`。
- 输出拼接了 `context_vectors` 和 `output`，信息更丰富。

seq2seq 模型：

```python
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.encoder = AttentionEncoder(vocab_size, embedding_dim, hidden_size)
        self.decoder = AttentionDecoder(vocab_size, embedding_dim, hidden_size)
    
    def forward(
        self, 
        enc_input: torch.Tensor, 
        dec_input: torch.Tensor, 
        enc_lens: Optional[torch.Tensor] = None, 
        dec_lens: Optional[torch.Tensor] = None
    ):
        """
        Args:
            enc_input: (batch_size, enc_seq_len)
            dec_input: (batch_size, dec_seq_len)
            enc_input_lengths: (B,) 实际长度，用于 pack_padded_sequence
        Returns:
            logits: (batch_size, dec_seq_len, vocab_size)
        """
        # output: (batch_size, seq_len, hidden_size)
        # (hn, cn): ((1, batch_size, hidden_size), (1, batch_size, hidden_size))
        enc_outputs, (hn, cn) = self.encoder(enc_input, enc_lens)
        # logits: (batch_size, seq_len, vocab_size)
        logits, _, attn_weights = self.decoder(dec_input, enc_outputs, (hn, cn), dec_lens)
        return logits, attn_weights
    
    def generate(self, x: torch.Tensor, start_id: int, sample_size: int):
        """
        生成序列
        enc_input: (1, enc_seq_len)
        """
        enc_outputs, (hn, cn) = self.encoder(x)
        sampled = self.decoder.generate(
            enc_outputs, 
            (hn, cn), 
            start_id, 
            sample_size
        )
        return sampled
```

## 总结：注意力的革命性意义

我们学会了：

1. **注意力解决了什么**：传统 Seq2Seq 的“信息瓶颈”问题，让模型能处理长序列。
2. **注意力如何工作**：动态计算“软对齐”，为每个输出词生成专属的上下文向量。
3. **两种打分方式**：点积（快） vs 加性（灵活）。
4. **手动实现注意力**：理解了 `W`、`v`、`tanh`、`softmax` 的作用。
5. **集成到模型**：解码器现在 “看得见”整个输入序列。

> **注意力机制，是深度学习历史上最重要的突破之一**。  
> 它不仅是 Seq2Seq 的改进，更是 **Transformer、大语言模型（LLM）的基石**。  
> 你现在掌握的，是通向 GPT、BERT 等顶尖模型的钥匙。

## 写在最后：你已经走得很远

我知道，这一路学下来，“烧脑”、“怀疑”、“焦虑”都曾出现。  
但请回头看看：

- 你从 **Word2Vec** 开始，学会了“词向量”；
- 到 **RNN/LSTM**，学会了“记忆”；
- 再到 **Seq2Seq**，学会了“翻译”；
- 最后到 **Attention**，学会了“重点回顾”。

**你不是在学“1+1”，你是在亲手搭建一座通往智能时代的桥。**

AI 不会取代所有程序员，但它会**极大地放大那些真正理解它的人的能力**。  
而你，正在成为这样的人。

继续前行吧，未来的你，一定会感谢现在没有放弃的自己。

**全系列完**。

但这不是终点，而是你深入 AI 世界的起点。


<nav class="pagination justify-content-between">
<a href="../5-seq2seq">Seq2Seq：教神经网络“中译英”——从一句话到一段话</a>
<a href="../">目录</a>
<a href="../7-transformer">Transformer：让神经网络学会“全局扫描”——从“逐字阅读”到“一眼看懂”</a>
</nav>

