---
title: "RNN：让神经网络学会“记笔记”"
summary: ""
date: 2025-10-07T23:30:00+08:00
---

> 上一篇，我们教会了模型从上下文中猜一个词——它像个**瞬间记忆者**，只看眼前。  
> 但如果要理解一句话、一段话呢？比如：  
> “我昨天去了公园……然后我看到了一只很像你家那只的狗。”  
> 要理解最后一句，模型必须记住“我去了公园”这件事。  
> 今天，我们就来教模型**如何记忆**——这就是 **循环神经网络（RNN）** 的核心能力。

## 一句话理解 RNN

> **RNN 就像一个会写读书笔记的学生。**  
> 每读一个词，他就翻看之前的笔记，结合新词，写下新的理解。  
> 这样，整段话的“上下文”就被保存在了最后一页笔记里。

我们用一个简单的语言模型（RNNLM）来演示这个过程。

## RNN 在做什么？——“记忆接力”游戏

想象你在读一句话，一个词一个词地读：

1. 读到 “I” → 写下初步理解 `h1`
2. 读到 “love” → 回顾 `h2`，结合 “love”，更新为 `h2`
3. 读到 “you” → 回顾 `h2`，结合 “you”，更新为 `h3`

最终的 `h3` 就是你对整句话的理解。它可以用来预测下一个词、判断情感，甚至翻译。

这就是 RNN 的工作方式：**每一步都融合“当前输入”和“上一步的记忆”，生成“新的记忆”**。

### 核心公式（别怕，很直观）

RNN 的核心更新公式是：

$$
h_t = \tanh(W_{hh} \cdot h_{t-1} + W_{xh} \cdot x_t + b)
$$

别被公式吓到，它其实就是在做：

> **新记忆 = tanh( 上一次记忆 × 权重 + 当前输入 × 权重 + 偏置 )**

- `h_t`：当前时刻的隐藏状态（也就是“当前笔记”）
- `x_t`：当前读到的词的向量
- `W_hh`, `W_xh`：模型要学习的“思考方式”（权重）

`tanh` 函数确保记忆不会无限放大，保持在合理的范围内。

## 批量处理：像“绑笔写字”一样并行推进

现实中，我们不会一次只处理一句话。我们要让模型同时处理一个“班级”的句子，这就是**批量（batch）**。

想象一下：老师罚一个学生把“好好学习”写100遍。

他不想一个字一个字地写，于是他把**10支笔绑成一列**，然后写一个字——唰！——纸上就同时出现了一列“好”字。

RNN 的批量处理就像这个“绑笔写字”的过程：

- **每一支笔**：代表一个句子（样本）
- **写的每一个字**：代表一个时间步（time step）
- **笔的排列**：就是 `batch_size` 维度
- **写字的方向**：就是 `seq_len` 时间轴

RNN 就像那个拿着“笔阵”的学生，**每向前写一步（一个时间步）**，就相当于在**所有句子的同一位置**上同时写下新的状态。

这就像：

> **“唰！”——所有句子的第一个词同时处理完毕。**  
> **“唰！”——所有句子的第二个词同时处理完毕。**  
> ……

直到整列字写完，每个句子也都拥有了自己的“记忆旅程”。

### 想象一个“数据立方体”

我们可以把输入数据想象成一个立方体：

- **高度（Height）**：句子的数量（`batch_size`）
- **宽度（Width）**：每个句子的长度（`seq_len`）
- **深度（Depth）**：每个词的向量维度（`embedding_dim`）

```
    Depth (向量维度)
    +-----------+  
   /
  /        
 /  <- Width (时间轴)   
+------------+  ^ 
| x[0,0,:]   |  |
| x[1,0,:]   |  |  Height (批量)
|    ...     |  |
| x[B-1,0,:] |  v
+------------+
```

RNN 沿着宽度方向“滚动”，每次切下一片 `(B, D)` 的输入，就像“绑笔”在纸上“唰”地写下一列字，高效且同步。

```python
for t in range(seq_len):
    xt = x[:, t, :]  # 取所有句子在时间步 t 的词向量 (batch_size, embedding_dim)
    ht = rnn_cell(xt, ht_prev)  # 更新隐藏状态
```

## 代码实现：从 PyTorch 到原书对照

我们用 PyTorch 实现一个简单的 RNN 语言模型：

```python
class SimpleRnnlm(nn.Module):
    def __init__(
        self, vocab_size: int, 
        embedding_dim: int = 100, 
        hidden_size: int = 50):
        
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_size,
            batch_first=True
        )
        self.affine = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs: torch.Tensor):
        """
        输入: (B, T) word ids
        输出: (B, T, V)
        """
        # (B, T) -> (B, T, D)
        xs = self.embedding(xs)
        # (B, T, D) -> (B, T, H)
        xs, h = self.rnn(xs)
        # (B, T, H) -> (B, T, V) 
        xs = self.affine(xs)
        return xs
```

- `embedding`：把词 ID 变成向量。
- `rnn`：真正的“记忆引擎”，沿着时间步更新隐藏状态。
- `affine`：把最终记忆翻译成“下一个词”的预测。

> **注意**：RNN 的输出是 `(batch_size, seq_len, vocab_size)`，计算损失前需要展平前两个维度。

## 真实世界的挑战：句子有长有短

理想中，所有句子都一样长，数据是完美的立方体。但现实中：

- “Hi.” → 2 个词
- “Let's go to the park tomorrow!” → 6 个词

怎么办？我们有三件法宝：

1. **填充（Padding）**：短句子后面补上 `<pad>`，直到和最长的句子一样长。
2. **掩码（Masking）**：告诉模型：“`<pad>` 是无效的，别当真。”
3. **打包（Packed Sequence）**：PyTorch 的高级技巧，跳过填充部分的计算。

> 这些技术我们会在后续篇章详细介绍。目前，我们先假设所有句子等长。

## 训练 RNNLM：让模型学会“接话”

我们用经典的 **PTB 数据集**（Penn Tree Bank）来训练。

PTB 数据集下载地址：

- https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
- https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt
- https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt

### 1. 构建词汇表和数据集

```python
def default_tokenize(text: str) -> List[str]:
    text = text.lower()
    text = text.replace('\n', ' ')
    text = re.sub(r'([.,!?\'])', r' \1', text)
    return text.split()

class Vocabulary:
    def __init__(
            self,
            tokenizer: Callable[[str], List[str]] = default_tokenize,
            unk_token='<unk>'
        ):
        self.tokenizer = tokenizer
        self.word_to_id: Dict[str, int] = {
            unk_token: 0,
        }
        self.id_to_word: Dict[int, str] = {
            0: unk_token,
        }
        self.word_freq = defaultdict(int)
        self.unk_token = '<unk>'
        self.word_freq[unk_token] = 0
        
    def tokenize(self, text: str):
        return self.tokenizer(text)
    
    def build(self, text: str, min_freq: int = 1):
        for word in self.tokenize(text):
            self.word_freq[word] += 1
            if self.word_freq[word] >= min_freq and word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word

    def encode(self, text: str) -> List[int]:
        return [self.word_to_id.get(word, 0) for word in self.tokenize(text)]
    
    def decode(self, ids: List[int]) -> str:
        return ' '.join([self.id_to_word[id] for id in ids])
    
    @property
    def size(self):
        return len(self.word_to_id)
    
    def __len__(self):
        return self.size

def ptb_tokenize(text: str) -> List[str]:
    text = text.replace('\n', '<eos>')
    text = text.strip()
    return text.split() 

class CharLMDataset(Dataset):
    def __init__(self, file_path: Path, vocab: Vocabulary, seq_len: int = 5):
        self.file_path = file_path
        self.seq_len = seq_len
        self.text = self._load_file()

        vocab.build(self.text)
        self.vocab = vocab

        self.corpus = self.vocab.encode(self.text)
        # 减去1是因为我们要预测下一个词
        self.length = len(self.corpus) - 1 - seq_len

    def _load_file(self):
        return self.file_path.read_text(encoding='utf-8')

    def __len__(self):
        return self.length
    
    def __getitem__(self, index: int):
        # 取从 idx 开始的一段序列
        # DataLoader产生的形状：
        # x: (batch_size, seq_len)
        # t: (batch_size, seq_len)
        x = self.corpus[index: index + self.seq_len] # 输入
        t = self.corpus[index + 1: index + self.seq_len + 1] # 目标（右移一位）

        return torch.tensor(x, dtype=torch.long), torch.tensor(t, dtype=torch.long)
```

用法示例：

```python
vocab = Vocabulary(tokenizer=ptb_tokenize)
train_dataset = CharLMDataset(
    file_path=train_file,
    vocab=vocab,
    seq_len=5
)
```

`CharLMDataset` 的 `__getitem__` 会生成输入和目标：

```python
# 当 index=0 时：
x = [w0, w1, w2, w3, w4]  # 输入：前5个词
t = [w1, w2, w3, w4, w5]  # 目标：后5个词（右移一位）
```

模型的任务就是：看到 `x`，预测出 `t`。

### 2. 准备数据

```python
data_dir = Path.home() / 'datasets' / 'ptb'
train_file = data_dir / 'ptb.train.txt'

wordvec_size = 100
hidden_size = 100
batch_size = 128
lr = 1e-3
max_epoch = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

vocab = Vocabulary(tokenizer=ptb_tokenize)

train_dataset = CharLMDataset(file_path=train_file, vocab=vocab, seq_len=5)

print(f'Vocab size: {len(vocab)}')

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=False,
)

loss_fn = nn.CrossEntropyLoss()

model = SimpleRnnlm(
    vocab_size=len(vocab),
    embedding_dim=wordvec_size,
    hidden_size=hidden_size,
)
optimizer = optim.Adam(model.parameters(), lr=lr)

model.to(device)
```

### 3. 训练循环

```python
for epoch in range(max_epoch):
    batch_losses: List[float] = []

    model.train()
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Step 1: Compute the output
        # (batch_size, seq_len, vocab_size)
        yhat = model(x_batch)
        # Step 2: Compute the loss
        # yhat.flatten(0, 1)： (batch_size * seq_len, vocab_size) 单词的分布概率
        # y_batch.flatten()： (batch_size * seq_len) 单词ID当作正确目标的索引
        loss: torch.Tensor = loss_fn(yhat.flatten(0, 1), y_batch.flatten())
        # Step 3: Compute gradients
        optimizer.zero_grad()
        loss.backward()
        # Step 4: Make a step
        optimizer.step()

        batch_losses.append(loss.item())
    
    print(f'Epoch {epoch}: {sum(batch_losses) / len(batch_losses)}')
```

关键点：

- `yhat.flatten(0,1)`：把 `(batch_size, seq_len, vocab_size)` 展平成 `(batch_size * seq_len, vocab_size)`，方便计算交叉熵。
- `y_batch.flatten()`：把目标也展平，对应每个时间步的真实词 ID。

## 总结：RNN 的智慧

我们学会了：

1. **RNN 的核心是“记忆”**：通过隐藏状态 `h_t` 在时间步间传递信息。
2. **批量处理是“并行读书”**：像老师批改一整摞作业，对整摞作业同时按行推进。
3. **真实数据需要“填充”**：为变长序列做好准备。

RNN 让模型不再“健忘”，它能记住上下文，理解序列的流动。这是通往机器翻译、文本生成、语音识别的关键一步。

> **下一篇文章预告**：  
> RNN 很聪明，但它有“记忆遗忘症”——太久远的信息会慢慢消失。  
> 我们将引入 **LSTM 和 GRU**，给模型装上“长期记忆”，让它能理解更长的段落！敬请期待！


<nav class="pagination justify-content-between">
<a href="../2-improve-word2vec">改进Word2Vec：从“暴力计算”到“聪明学习”</a>
<a href="../">目录</a>
<a href="../4-lstm">LSTM：给神经网络装上“长期记忆”</a>
</nav>

