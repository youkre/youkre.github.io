---
title: "LSTM：给神经网络装上“长期记忆”"
summary: ""
date: 2025-10-08T08:05:00+08:00
---

> 上一篇，我们学会了 RNN——一个会写读书笔记的学生。  
> 但这个学生有个毛病：**记性不太好**。  
> 如果文章太长，他前面写的笔记就会慢慢模糊，甚至被新内容覆盖。  
> 比如读到《红楼梦》最后一回时，他已经忘了贾宝玉是谁了……  
>
> 今天，我们就来给他升级大脑，装上一个**不会遗忘的长期记忆系统**——这就是 **LSTM（长短期记忆网络）**。

## 一句话理解 LSTM

> **LSTM 就像 RNN 的超级升级版，它有两个脑子：**
> **一个“工作台”（隐藏状态 `h_t`）**：处理当前任务，和外界交流。
> **一个“保险柜”（细胞状态 `c_t`）**：专门存放重要、长期的信息，不受日常干扰。

RNN 只有一个笔记本，写满就擦掉重来。

而 LSTM 多了一个**带密码锁的保险柜**，只有特定的“守门人”才能决定往里存什么、取什么。这样，关键信息就不会丢失了。

## LSTM 的三大“守门人”

LSTM 的智慧在于它的**门控机制（Gating Mechanism）**。它有三位“守门人”，分别控制信息的流动：

1. **遗忘门（Forget Gate）**：决定**丢弃**保险柜里的哪些旧信息。
2. **输入门（Input Gate）**：决定**更新**哪些新信息进保险柜。
3. **输出门（Output Gate）**：决定从保险柜中**取出**哪些信息放到工作台上使用。

我们用一个例子来理解：

> 假设你在读一本侦探小说：
> 一开始你知道“凶手是张三” → 这个信息被锁进保险柜。
> 后来发现张三是无辜的 → **遗忘门**打开，把“张三是凶手”这条信息擦掉。
> 新线索指向李四 → **输入门**打开，把“李四是凶手”存进去。
> 到结尾要推理时 → **输出门**打开，把“李四是凶手”拿出来用。

这三位“守门人”都是由神经网络学习出来的，它们能自动判断什么信息重要、什么该丢弃。

### 详细工作流程

在每个时间步 `t`，LSTM 接收当前输入 `x_t` 和上一时刻的隐藏状态 `h_{t-1}`，然后：

#### 1. 遗忘门：该忘的就忘

```python
f_t = sigmoid(W_f @ [h_{t-1}, x_t] + b_f)
```

- 输出 `f_t` 是一个 0~1 之间的向量。
- `f_t[i] ≈ 0`：表示细胞状态 `c_{t-1}[i]` 的第 `i` 项信息应该被**完全遗忘**。
- `f_t[i] ≈ 1`：表示该项信息应该被**完全保留**。

> **直觉**：就像整理书架，把过时的书扔掉，留下经典。

#### 2. 输入门：该学的就学

包含两部分：

- **候选值 `c̃_t`**：可能要加入的新信息。

```python
c̃_t = tanh(W_c @ [h_{t-1}, x_t] + b_c)
```

- **输入门 `i_t`**：决定多少新信息可以进入。

```python
i_t = sigmoid(W_i @ [h_{t-1}, x_t] + b_i)
```

#### 3. 更新细胞状态

把旧保险柜的内容按 `f_t` 忘掉，再把新信息按 `i_t` 存入：

```python
c_t = f_t * c_{t-1} + i_t * c̃_t
```

> 这就是 LSTM 的核心！细胞状态 `c_t` 像一条“信息高速公路”，贯穿整个序列，只在门口有少量加减乘除操作，因此能很好地保留长期依赖。

#### 4 输出门：该用的才用

```python
o_t = sigmoid(W_o @ [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(c_t)
```

- `o_t` 决定从保险柜中取出哪些信息。
- `tanh(c_t)` 是保险柜内容的“净化版”。
- 最终的 `h_t` 是工作台上的当前记忆，用于预测或传给下一个时间步。

## PyTorch 实现：只需替换一行

好消息是，PyTorch 让我们不用手动实现这些复杂的门控逻辑。只需要把上一章 RNN 模型中的 `nn.RNN` 换成 `nn.LSTM`：

```python
class Rnnlm(nn.Module):
    def __init__(self, vocab_size=10000, embedding_dim=100, hidden_size=100) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)  # ← 只换这一行！
        self.affine = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs: torch.Tensor):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        xs = self.embedding(xs)
        # xs: (batch_size, seq_len, hidden_size)
        xs, (hn, cn) = self.lstm(xs)  # 返回隐藏状态和细胞状态
        # (batch_size, seq_len, vocab_size)
        xs = self.affine(xs)
        return xs
```

- `hn`：最终的隐藏状态 `(num_layers, batch_size, hidden_size)`
- `cn`：最终的细胞状态 `(num_layers, batch_size, hidden_size)`

在语言模型中，我们通常只关心 `xs`（每个时间步的输出），但 `hn` 和 `cn` 在 Seq2Seq 等任务中非常有用。

## 改善模型：堆叠与 Dropout

我们可以进一步提升 LSTM 的性能：

```python
class BetterRnnlm(nn.Module):
    def __init__(
        self, 
        vocab_size=10000, 
        embedding_dim=100, 
        hidden_size=100, 
        dropout_ratio=0.5):

        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.dropout1 = nn.Dropout(dropout_ratio)
        self.lstm1 = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        
        self.dropout2 = nn.Dropout(dropout_ratio)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.dropout3 = nn.Dropout(dropout_ratio)
        self.affine = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs: torch.Tensor):
        # (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        xs = self.embedding(xs)
        xs = self.dropout1(xs)
        # xs: (batch_size, seq_len, hidden_size)
        # hn: (1, batch_size, hidden_size)
        # cn: (1, batch_size, hidden_size)
        xs, (hn, cn) = self.lstm1(xs)
        
        xs = self.dropout2(xs)
        xs, (hn, cn) = self.lstm2(xs, (hn, cn))  # 注意：传入上一层的 hn, cn
        
        xs = self.dropout3(xs)
        # (batch_size, seq_len, vocab_size)
        xs = self.affine(xs)
        return xs
```

- **堆叠 LSTM**：两个 LSTM 层串联，形成更深的网络，能学习更复杂的模式。
- **Dropout**：防止过拟合，让模型更鲁棒。

## 训练

数据准备和训练步骤于上一章完全一样，只需要替换模型即可。

不过这里我们稍微改动一下，以便为下一章生成文本作准备。对 `VOcabulary` 类添加 `save` 和 `load` 方法，以前复用之前的词汇表：

```python
class Vocabulary:
    # 其他部分同上一章定义的部分

    def save(self, path: Path):
        # Do not save id_to_word as int cannot be used as key in json.
        vocab = {
            'word_to_id': self.word_to_id,
            'word_freq': self.word_freq,
            'unk_token': self.unk_token,
        }
        path.mkdir(parents=True, exist_ok=True)
        data = json.dumps(
            vocab,
            ensure_ascii=False, 
            indent=4,
        )
        filepath = path.joinpath('vocab.json')
        filepath.write_text(data)
        print(f'Vocabulary saved to {filepath}')

    def load(self, path: Path):
        with path.joinpath('vocab.json').open('r', encoding='utf-8') as f:
            vocab = json.load(f)
            
        self.word_to_id = vocab['word_to_id']
        self.id_to_word = {
            v: k
            for k, v in self.word_to_id.items()
        }
        self.word_freq = vocab['word_freq']
        self.unk_token = vocab['unk_token']
```

训练完成后保存词汇表和模型：

```python
out_dir = Path.home() / 'datasets' / 'better_rnnlm'
state_file = out_dir.joinpath('better_rnnlm.pth')
vocab.save(out_dir)
torch.save(model.state_dict(), state_file)
```

## 总结：LSTM 的智慧

我们学会了：

1. **LSTM 解决了 RNN 的“健忘症”**：通过细胞状态 `c_t` 保存长期信息。
2. **三大门控机制是关键**：
    - 遗忘门 → 忘掉无关信息
    - 输入门 → 记住重要新知
    - 输出门 → 按需提取记忆
3. **PyTorch 使用极其简单**：只需替换 `nn.RNN` 为 `nn.LSTM`。
4. **堆叠与 Dropout 提升性能**：构建更深、更稳健的模型。

LSTM 让模型不仅能记住上下文，还能分辨**什么是重要的、什么是次要的**。这是迈向真正“理解”语言的重要一步。


<nav class="pagination justify-content-between">
<a href="../3-rnn">RNN：让神经网络学会“记笔记”</a>
<a href="../">目录</a>
<span></span>
</nav>

