---
title: "Transformer：让神经网络学会“全局扫描”——从“逐字阅读”到“一眼看懂”"
summary: "使用自注意力机制实现编码器-解码器模型"
date: 2025-10-29T23:32:00+08:00
---

> 在上一篇《注意力机制》中，我们教会了神经网络“重点回顾”——在翻译或生成时，动态关注输入句子中最相关的词。这就像一个阅卷老师，一边读你的答案，一边回头对照题目，判断你答得对不对。
>
> 但你还记得吗？那个模型的“阅读方式”(即 RNN 层)依然是**线性的**——它像人一样，一个字一个字地读完输入，再一个字一个字地写出输出。这种“逐字阅读”的方式，不仅慢，而且容易“忘记开头”。
>
> 今天，我们要迈出最关键的一步：**彻底抛弃RNN，让模型学会“一眼看懂”整句话**。
>
> 这就是——**Transformer**。

## 一、我们想要的“阅读方式”是什么？

想象一下，你看到一句话：

> “我喜欢吃苹果，因为它很甜。”

当你读到“它”时，你**瞬间**就知道“它”指的是“苹果”，而不是“我”或“吃”。你不需要从头再读一遍，也不需要像RNN那样一步步传递状态。

你是怎么做到的？因为你**同时看到了整句话的所有词**，并快速建立了它们之间的联系。

这就是我们想要的模型能力：**并行地、全局地理解一个句子中所有词之间的关系**。

而实现这一点的核心技术，就是——**自注意力机制（Self-Attention）**。

## 二、自注意力：让每个词“看见”所有词

“自注意力”这个名字听起来很玄，其实很简单：

> **让句子中的每一个词，都去“问”其他所有词：“你们和我有什么关系？”**

我们以一个简单的例子来说明。假设输入序列是：

```python
['a', 'b', 'c', 'd']
```

我们想计算 `b` 的“上下文向量”——也就是融合了整句话信息后，`b` 应该变成什么样子。

### 第一步：计算“相关性分数”

模型会为每个词生成三个向量：**Query（查询）**、**Key（键）**、**Value（值）**。

- **Query**：代表“我现在在找什么？”
- **Key**：代表“我这个位置能提供什么？”
- **Value**：代表“我这个位置的实际内容是什么？”

> 小贴士：你可以把它们想象成图书馆的索引系统：
>
> - Query 是你在搜索框里输入的关键词；
> - Key 是每本书的标签；
> - Value 是书本身的内容。

计算 `b` 和其他词的相关性，就是用 `b` 的 **Query** 去和所有词的 **Key** 做点积：

```text
score(b,a) = Query_b · Key_a
score(b,b) = Query_b · Key_b
score(b,c) = Query_b · Key_c
score(b,d) = Query_b · Key_d
```

这个分数越高，说明那个词和 `b` 越相关。

如果我们把所有词的 Query 和所有词的 Key 做点积，就得到一个 **注意力分数矩阵**：

```text
      a    b    c    d
a   0.1  0.2  0.3  0.4
b   0.5  0.6  0.7  0.8
c   0.9  0.8  0.7  0.6
d   0.5  0.4  0.3  0.2
```

这个矩阵告诉我们：每个词对其他词的“关注度”。

### 第二步：归一化——变成“概率”

分数不能直接用，我们要把它变成加起来为1的概率分布。用 **Softmax**：

```python
weights = softmax(scores)
```

比如 `b` 这一行变成：

```text
[0.1, 0.2, 0.3, 0.4] → [0.18, 0.24, 0.30, 0.28]
```

这意味着：`b` 最关注 `c`，其次是 `b` 自己，然后是 `d` 和 `a`。

### 第三步：加权求和——生成“上下文向量”

最后，用这些权重去加权所有词的 **Value** 向量：

```python
context_b = 0.18*Value_a + 0.24*Value_b + 0.30*Value_c + 0.28*Value_d
```

这个 `context_b` 就是 `b` 在全局上下文中的新表示——它融合了整句话的信息。

**关键点**：这个过程对每个词都独立进行，所以可以**完全并行计算**！不像RNN必须等前一个词算完。

## 三、多头注意力：让模型“多角度看问题”

如果只用一组 Query、Key、Value，模型的“注意力”可能不够丰富。

就像一个人看问题可能片面，但我们如果让**多个专家**同时看，就能得到更全面的判断。

这就是 **多头注意力（Multi-Head Attention）**。

- 我们让模型训练 **多个** 独立的注意力头。
- 每个头学习不同的 Query、Key、Value 变换。
- 最后把所有头的输出拼在一起，再投影回原始维度。

这样，有的头可能关注语法结构，有的头关注语义关系，有的头关注指代关系……模型的“理解力”就更强了。

## 四、手写一个 Transformer 模块

现在我们用 PyTorch 实现一个完整的 **多头自注意力层**。

```python
import torch
import torch.nn as nn
from typing import Optional

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,      # 词向量维度，如 768
        num_heads: int,    # 注意力头数，如 12
        context_length: int, # 上下文长度，如 512
        dropout: float = 0.1,
        qkv_bias: bool = False
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 每个头的维度

        # 线性变换：将输入映射到 Q, K, V
        self.W_query = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_key   = nn.Linear(d_model, d_model, bias=qkv_bias)
        self.W_value = nn.Linear(d_model, d_model, bias=qkv_bias)

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_model)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 因果掩码（用于解码器，防止看到未来）
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1).bool()
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False
    ):
        """
        query: (batch_size, seq_len_q, d_model)
        key:   (batch_size, seq_len_k, d_model)
        value: (batch_size, seq_len_v, d_model)
        key_padding_mask: (batch_size, seq_len_k)  # 填充位置为 True
        is_causal: 是否使用因果掩码（如解码器自注意力）
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # 1. 线性变换得到 Q, K, V
        queries = self.W_query(query)  # (B, T_q, D)
        keys    = self.W_key(key)      # (B, T_k, D)
        values  = self.W_value(value)  # (B, T_v, D)

        # 2. 拆分成多个头
        # (B, T_q, D) -> (B, T_q, num_heads, head_dim)
        queries = queries.view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        keys    = keys.view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        values  = values.view(batch_size, seq_len_k, self.num_heads, self.head_dim)

        # 3. 调整维度：把头放到第2维
        # (B, T_q, H, D_h) -> (B, H, T_q, D_h)
        queries = queries.transpose(1, 2)
        keys    = keys.transpose(1, 2)
        values  = values.transpose(1, 2)

        # 4. 计算注意力分数
        # (B, H, T_q, D_h) @ (B, H, D_h, T_k) -> (B, H, T_q, T_k)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)  # 缩放

        # 5. 应用掩码
        if key_padding_mask is not None:
            # (B, T_k) -> (B, 1, 1, T_k)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, -torch.inf)

        if is_causal:
            # 动态截取因果掩码
            causal_mask = self.causal_mask[:seq_len_q, :seq_len_k].to(attn_scores.device)
            attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)

        # 6. Softmax 归一化
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 7. 加权求和
        # (B, H, T_q, T_k) @ (B, H, T_k, D_h) -> (B, H, T_q, D_h)
        context_vec = torch.matmul(attn_weights, values)

        # 8. 合并所有头
        # (B, H, T_q, D_h) -> (B, T_q, H, D_h) -> (B, T_q, D)
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch_size, seq_len_q, self.d_model)

        # 9. 输出投影
        context_vec = self.out_proj(context_vec)
        return context_vec
```

> **注意**：这个实现支持 **自注意力**（`query=key=value`）和 **交叉注意力**（`query`来自解码器，`key/value`来自编码器），非常灵活。

## 五、构建 Transformer 编码器

现在我们用 `MultiHeadAttention` 构建一个 **Transformer 编码器块**。

每个块包含：

- 多头自注意力
- 前馈神经网络（FFN）
- 层归一化（LayerNorm）
- 残差连接

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False

class LayerNorm(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

class FeedForward(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            GELU(),
            nn.Linear(4 * embedding_dim, embedding_dim)
        )

    def forward(self, x):
        return self.layers(x)

class EncoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_model=cfg.emb_dim,
            num_heads=cfg.n_heads,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ff = FeedForward(embedding_dim=cfg.emb_dim)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x, key_padding_mask=None):
        # 自注意力 + 残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.attn(x, x, x, key_padding_mask=key_padding_mask, is_causal=False)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈网络 + 残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

## 六、构建 Transformer 解码器

解码器比编码器多一个 **交叉注意力** 层，用来关注编码器的输出。

```python
class CrossAttention(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, context_length: int):
        super().__init__()
        self.mha = MultiHeadAttention(
            d_model=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
        )

    def forward(self, dec_hidden, memory, memory_key_padding_mask=None):
        return self.mha(
            query=dec_hidden,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            is_causal=False
        )

class DecoderBlock(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.sa = MultiHeadAttention(
            d_model=cfg.emb_dim,
            num_heads=cfg.n_heads,
            context_length=cfg.context_length,
            dropout=cfg.drop_rate,
            qkv_bias=cfg.qkv_bias,
        )
        self.ca = CrossAttention(
            embedding_dim=cfg.emb_dim,
            num_heads=cfg.n_heads,
            context_length=cfg.context_length,
        )
        self.ff = FeedForward(embedding_dim=cfg.emb_dim)
        self.norm1 = LayerNorm(cfg.emb_dim)
        self.norm2 = LayerNorm(cfg.emb_dim)
        self.norm3 = LayerNorm(cfg.emb_dim)
        self.drop_shortcut = nn.Dropout(cfg.drop_rate)

    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # 自注意力（带因果掩码）
        shortcut = x
        x = self.norm1(x)
        x = self.sa(x, x, x, key_padding_mask=tgt_key_padding_mask, is_causal=True)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 交叉注意力
        shortcut = x
        x = self.norm2(x)
        x = self.ca(x, memory, memory_key_padding_mask)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # 前馈网络
        shortcut = x
        x = self.norm3(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
```

## 七、组合成完整的 Seq2Seq 模型

最后，把编码器和解码器组装起来：

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)

    def forward(self, xs, key_padding_mask=None):
        seq_len = xs.shape[1]
        tok_embeds = self.tok_emb(xs)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=xs.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x, key_padding_mask)
        x = self.final_norm(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, vocab_size, bias=False)

    def forward(self, xs, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        seq_len = xs.shape[1]
        tok_embeds = self.tok_emb(xs)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=xs.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        for block in self.trf_blocks:
            x = block(x, memory, tgt_key_padding_mask, memory_key_padding_mask)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, cfg: ModelConfig):
        super().__init__()
        self.encoder = TransformerEncoder(vocab_size, cfg)
        self.decoder = TransformerDecoder(vocab_size, cfg)

    def forward(self, enc_inputs, dec_inputs, src_key_padding_mask=None, tgt_key_padding_mask=None):
        enc_outputs = self.encoder(enc_inputs, src_key_padding_mask)
        logits = self.decoder(dec_inputs, enc_outputs, tgt_key_padding_mask, src_key_padding_mask)
        return logits
```

## 训练

接下来准备训练数据，使用《深度学习进阶：自然语言处理》一书中的日期格式化数据集，可以在图灵官网 https://m.ituring.com.cn/book/2678 随书下载部分找到。

数据格式如下：

```text
september 27, 1994           _1994-09-27
August 19, 2003              _2003-08-19
2/10/93                      _1993-02-10
10/31/90                     _1990-10-31
TUESDAY, SEPTEMBER 25, 1984  _1984-09-25
JUN 17, 2013                 _2013-06-17
```

左侧部分作为编码器的输入，右侧部分作为解码器的输入。目标是是输入任意格式的日期，模型可以转换为标准格式。

定义一个分词器，因为这里的数据很简单，所以直接用字母分词：

```python
class BaseTokenizer:
    PAT_TOKEN= '<pad>'
    START_TOKEN = '<start>'
    END_TOKEN = '<end>'
    UNK_TOKEN = '<unk>'

    @property
    def pad_id(self):
        return self.encode(self.PAT_TOKEN)[0]
    
    @property
    def unk_id(self):
        return self.encode(self.UNK_TOKEN)[0]
    
    @property
    def start_id(self):
        return self.encode(self.START_TOKEN)[0]
    
    @property
    def end_id(self):
        return self.encode(self.END_TOKEN)[0]
    
    @abstractmethod
    def encode(self,
               text: str, 
               add_bos=False,
               add_eos=False) -> List[int]:
        pass

    def encode_to_tensor(self,
                         text: str,
                         add_bos=False,
                         add_eos=False):
        encoded = self.encode(text, add_bos, add_eos)
        # Add batch dimension
        encoded_tensor = torch.tensor(encoded).unsqueeze(0)
        return encoded_tensor

    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        pass

    def decode_from_tensor(self, token_ids: torch.Tensor):
        # Remove batch dimension
        flat = token_ids.squeeze(0)
        return self.decode(flat.tolist())
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        pass

class DateTokenizer(BaseTokenizer):
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}

    @property
    def pad_id(self):
        return self.word_to_id[self.PAT_TOKEN]
    
    @property
    def unk_id(self):
        return self.word_to_id[self.UNK_TOKEN]
    
    @property
    def start_id(self):
        return self.word_to_id[self.START_TOKEN]

    def add_special_tokens(self):
        for token in [self.PAT_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
            self.add(token)


    def add(self, word: str):
        if word not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    def encode(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        tokens = [
            self.word_to_id.get(c, self.unk_id) 
            for c in text
        ]
        if add_bos:
            tokens = [self.start_id] + tokens
        if add_eos:
            tokens += [self.end_id]

        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        words = []
        for idx in token_ids:
            word = self.id_to_word.get(idx, self.UNK_TOKEN)
            if word == self.END_TOKEN:
                break
            if word not in [self.PAT_TOKEN, self.START_TOKEN]:
                words.append(word)

        return ''.join(words)
    
    def get_vocab_size(self) -> int:
        return len(self.word_to_id)
```

然后定义数据集：

```python

@dataclass
class Seq2SeqRaw:
    source: str
    target: str

@dataclass
class Seq2SeqBatchItem:
    source_ids: List[int]
    target_ids: List[int]

class DateDataset(Dataset):
    def __init__(self, corpus: List[Seq2SeqRaw], tokenizer: BaseTokenizer):
        self.corpus = corpus
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx: int) -> Seq2SeqBatchItem:
        entry = self.corpus[idx]

        encoder_input = self.tokenizer.encode(entry.source)
        decoder_input = self.tokenizer.encode(entry.target)

        return Seq2SeqBatchItem(
            source_ids=encoder_input,
            target_ids=decoder_input,
        )
```

自定义一个 collate 函数来处理批次的填充、生成目标序列等。注意这里 `pad_sequence` 来自 `rnn` 包，不过它是通用的，并不尽限于 RNN，但是另外两个函数 `pack_padded_sequence` 和 `pad_packed_sequence` 就是针对 RNN 结构的，所以并不适用于其他模型。这里我们在填充序列的同时创建掩码，作为 `MultiHeadAttention` 中的`key_padding_mask` 参数：

```python
from torch.nn.utils.rnn import pad_sequence

def seq2seq_collate_fn(
        batch: List[Seq2SeqBatchItem],
        tokenizer: BaseTokenizer,
        device: torch.device):
    
    sources = [item.source_ids for item in batch]
    targets = [item.target_ids for item in batch]

    enc_inputs = pad_sequence(
        [torch.tensor(s) for s in sources],
        batch_first=True,
        padding_value=tokenizer.pad_id,
    )

    enc_mask = (enc_inputs == tokenizer.pad_id).bool()

    dec_inputs = pad_sequence(
        [
            torch.tensor([tokenizer.start_id] + t)
            for t in targets
        ],
        batch_first=True,
        padding_value=tokenizer.pad_id,
    )

    dec_mask = (dec_inputs == tokenizer.pad_id).bool()

    targets_tensor = pad_sequence(
        [
            torch.tensor(t + [tokenizer.end_id])
            for t in targets
        ],
        batch_first=True,
        padding_value=-100, # -100 是 ignore_index
    )

    return Seq2SeqBatch(
        encoder_input=enc_inputs.to(device),
        encoder_mask=enc_mask.to(device),
        decoder_input=dec_inputs.to(device),
        decoder_mask=dec_mask.to(device),
        targets=targets_tensor.to(device),
    )
```

注意这里目标序列需要相对于解码器输入左移 1 位，也就是解码器每个位置的输出是为了预测下一个位置。不过这里我们在解码器输入的开始位置填充了一个开始标记，标签序列的结尾则加了一个结束标记，这样就实现了错开一位的目的。`padding_value=-100` 则是交叉熵误差默认忽略的索引。

为了方便管理数据创建一个辅助类：

```python
class DateDataBuilder:
    def __init__(self, file_path: Path) -> None:
        self.data = self.load(file_path)

    def load(self, file_path: Path):
        if not file_path.exists():
            raise FileNotFoundError(f"{file_path} does not exist")
        
        pairs: List[Seq2SeqRaw] = []
        for line in file_path.open(encoding="utf-8"):
            src, tgt = line.strip().split('_')
            pairs.append(
                Seq2SeqRaw(
                    source=src.strip(),
                    target=tgt.strip()
                )
            )

        return pairs
    
    def create_tokenizer(self):
        chars = set()
        for entry in self.data:
            chars.update(entry.source)
            chars.update(entry.target)

        tokenizer = DateTokenizer()
        for char in chars:
            tokenizer.add(char)

        tokenizer.add_special_tokens()
        return tokenizer
    
    def split_data(self, train_frac: float = 0.85, test_frac: float = 0.1):
        train_portion = int(len(self.data) * train_frac)
        test_portion = int(len(self.data) * test_frac)

        train_data = self.data[:train_portion]
        test_data = self.data[train_portion:train_portion + test_portion]
        val_data = self.data[train_portion + test_portion:]

        return train_data, test_data, val_data
```

由于数据量很少（5万条），所以不需要设置成像 GPT 那样大的规模，因此使用这个配置：

```python
@dataclass(frozen=True)
class LearningConfig(ModelConfig):
    context_length: int = 64
    emb_dim: int = 128
    n_heads: int = 4
    n_layers: int = 1
```

这个配置只使用 1 层 Transformer，参数量约 50 万，对于我们这个规模的数据足够了。

配置超参数：

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    return device

device = get_device()

date_builder = DateDataBuilder(date_file)

tokenizer = date_builder.create_tokenizer()

train_data, val_data, test_data = date_builder.split_data()

customized_collate_fn = partial(
    seq2seq_collate_fn,
    tokenizer=tokenizer,
    device=device,
)

num_epochs = 5
batch_size = 128
lr = 1e-4
tiny_config = LearningConfig()

model = Seq2Seq(
    vocab_size=tokenizer.get_vocab_size(),
    cfg=tiny_config
)

optimizer = optim.AdamW(
    model.parameters(), 
    lr=lr, 
    weight_decay=0.01
)

train_loader: DataLoader[Seq2SeqBatch] = DataLoader(
    DateDataset(train_data, tokenizer),
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
)

val_loader: DataLoader[Seq2SeqBatch] = DataLoader(
    DateDataset(val_data, tokenizer),
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
)

model.to(device)
```

在训练之前可以看看没有训练的模型会输出什么，这里还需要定义一个文本生成函数：

```python
def generate_text(
        model: nn.Module,
        tokenizer: BaseTokenizer,
        sample_text: str,
        max_new_tokens: int,
        context_size: int,
        device: torch.device,
        temperature=0.0,
        top_k: Optional[int] = None):
    
    # 编码输入
    # (1, seq_len)
    enc_tensor = tokenizer.encode_to_tensor(sample_text).to(device)
    
    # 开始生成
    generated = torch.tensor([[tokenizer.start_id]])
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            dec_tensor = generated[:, -context_size:].to(device)
        
       
            # (1, seq_len, vocab_size)
            logits = model(enc_tensor, dec_tensor)
            # (1, vocab_size)
            logits = logits[:, -1, :]
            if top_k is not None:
                top_logits, _ = torch.topk(logits, top_k)
                min_val = top_logits[:, -1]
                logits = torch.where(
                    logits < min_val,
                    torch.tensor(float('-inf')).to(logits.device),
                    logits,
                )

            if temperature > 0:
                logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            if idx_next == tokenizer.end_id:
                break
            
            generated = torch.cat((generated, idx_next), dim=1)
            
    
    final_text = tokenizer.decode_from_tensor(generated[:, 1:])  # 去掉BOS
    return final_text
```

这里可以选择使用温度缩放和 Top-k 采样，默认使用贪婪解码。

```python
def generate_and_print_sample(
        model: nn.Module, 
        tokenizer: BaseTokenizer,
        start_context: str,
        context_length: int,
        device: torch.device, 
    ):
    
    model.eval()

    decoded_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        sample_text=start_context,
        max_new_tokens=50,
        context_size=context_length,
        device=device
    )
    print(decoded_text.replace("\n", " "))
    model.train()

for entry in test_data[:10]:
    print(f">> Source: {entry.source}")
    generate_and_print_sample(
        model,
        tokenizer=tokenizer,
        start_context=entry.source,
        context_length=tiny_config.context_length,
        device=device,
    )
```

你会发现生成的结果大概率都是毫无意义的字母组合。

接下来开始训练：

```python
def train_seq2seq(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_epochs: int,
        eval_freq: int,
        eval_iter: int,
        start_context: str,
        tokenizer: BaseTokenizer,
        context_length: int
):
    train_losses, val_losses = [], []
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(
                model=model,
                batch=batch
            )
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    eval_iter=eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Epoch {epoch+1}, step {global_step:06d}: train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        generate_and_print_sample(
            model=model,
            tokenizer=tokenizer,
            device=device,
            start_context=start_context,
            context_length=context_length
        )
    
    return train_losses, val_losses

train_losses, val_losses = train_seq2seq(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=32,
    eval_iter=10,
    start_context="TUESDAY, SEPTEMBER 10, 1991",
    tokenizer=tokenizer,
    context_length=tiny_config.context_length
)
```

这里的模型和数据集都不大，在普通的 CPU 上几分钟就可以运行完。

这时候在用测试集试一下模型的输出，应该能正确转换了：

```python
model.eval()
for entry in test_data:
    input_text = entry.source
    
    encoded = tokenizer.encode_to_tensor(input_text)
    with torch.no_grad():
        generated = generate_text(
            model=model,
            tokenizer=tokenizer,
            sample_text=input_text,
            max_new_tokens=50,
            context_size=tiny_config.context_length,
            device=device,
        )

        print(f"{entry.source:<30} -> {generated} | Expected: {entry.target}")
```

这是部分输出结果：

```text
1/4/04                         -> 2004-01-04 | Expected: 2004-01-04
Sunday, August 8, 2010         -> 2010-08-08 | Expected: 2010-08-08
Jan 17, 1985                   -> 1985-01-17 | Expected: 1985-01-17
October 19, 1986               -> 1986-10-19 | Expected: 1986-10-19
october 31, 1998               -> 1998-10-31 | Expected: 1998-10-31
5/27/98                        -> 1998-05-27 | Expected: 1998-05-27
Thursday, July 24, 2003        -> 2003-07-24 | Expected: 2003-07-24
```

---

**参考资料：**

- 斋藤康毅《深度学习进阶：自然语言处理》
- 塞巴斯蒂安·拉施卡《从零构建大模型》


<nav class="pagination justify-content-between">
<a href="../6-attention">注意力机制：让神经网络学会“重点回顾”</a>
<a href="../">目录</a>
<span></span>
</nav>

