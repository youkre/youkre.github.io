---
title: "自然语言处理入门：从一句话到词向量——用PyTorch实现Word2Vec"
summary: "计算机不懂人类语言，它只懂数字。我们要让AI理解“猫”和“狗”是相似的动物，第一步就是把“猫”变成一串数字向量——这就是Word2Vec的核心思想。"
date: 2025-09-27T18:22:00+08:00
---

> 注：本系列是学习斋藤康毅《深度学习进阶：自然语言处理》时用PyTorch实现的笔记。

## 引言：大模型是怎么“看”语言的？

你有没有好奇过，像ChatGPT这样的大模型，是怎么理解我们说的“今天天气真好”这句话的？它真的像人一样“听懂”了吗？

答案是：**不，它并没有“听懂”，但它学会了用数学的方式“感知”语言。**

大模型处理语言的第一步，不是理解语义，而是**把文字变成数字**。因为计算机只能处理数字，不能直接理解“爱”、“学习”或“猫”这些词的含义。

那么问题来了：

- 怎么把一个词变成数字？
- 变成什么样的数字，才能让模型知道“国王”和“王后”是相关的？
- 为什么现在的AI能写出通顺的文章、回答复杂的问题？

本文将带你从零开始，一步步实现一个经典的自然语言处理模型——**Word2Vec**，并用PyTorch完成代码实现。即使你不是程序员或AI专家，也能看懂这个“语言数字化”的起点。

## 第一步：把句子拆成单词——分词（Tokenization）

假设我们有一句话：

> “Hello world. Hello NLP. NLP is fun!”

计算机要处理这句话，首先要把它“切开”，就像切蛋糕一样，切成最小的语言单位——**词（token）**。

我们写一个简单的切词函数：

```python
def default_tokenize(text: str) -> List[str]:
    text = text.lower()                    # 全部转小写
    text = text.replace('\n', ' ')         # 换行符换成空格
    text = re.sub(r'([.,!?\'])', r' \1', text)  # 标点符号前后加空格
    return text.split()                    # 按空格切分
```

运行后得到：

```text
['hello', 'world', '.', 'hello', 'nlp', '.', 'nlp', 'is', 'fun', '!']
```

现在，句子变成了一个**单词列表**，这是第一步。

## 第二步：给每个词分配一个“身份证号”——构建词汇表（Vocabulary）

接下来，我们要给每个词发一个唯一的“身份证号”——也就是**词ID**。

比如：

- `hello` → 1
- `world` → 2
- `.` → 3
- `nlp` → 4
- ...

我们用一个叫 `Vocabulary` 的类来管理这个映射关系：

```python
class Vocabulary:
    """
    在 Vocabulary 中传入 default_tokenize 函数，
    使用时可以随意替换。
    """
    def __init__(
            self,
            tokenizer: Callable[[str], List[str]] = default_tokenize,
            unk_token='<unk>'
        ):
        self.tokenizer = tokenizer
        # 单词到ID的映射，用于编码
        self.word_to_id: Dict[str, int] = {
            unk_token: 0,
        }
        # ID到单词的映射，用来解码
        self.id_to_word: Dict[int, str] = {
            0: unk_token,
        }
        self.word_freq = defaultdict(int)
        self.unk_token = '<unk>'
        self.word_freq[unk_token] = 0

    def build_from_text(self, text: str, min_freq: int = 1):
        """
        text: 输入的自然语言文本
        min_frq: 支持过滤低频词
        """
        for word in self.tokenizer(text):
            self.word_freq[word] += 1
            if self.word_freq[word] >= min_freq and word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word

    def add_special_token(self, token: str):
        """
        添加特殊 token，比如 <sos> <eos> <pad>
        """
        if token not in self.word_to_id:
            new_id = len(self.word_to_id)
            self.word_to_id[token] = new_id
            self.id_to_word[new_id] = token
            self.word_freq[token] = 0
    
    def encode(self, text: str) -> List[int]:
        """
        输入一段文本，转换为数字
        """
        return [self.word_to_id.get(word, 0) for word in self.tokenizer(text)]
    
    def decode(self, ids: List[int]) -> str:
        """
        把数字转换回自然语言
        """
        return ' '.join([self.id_to_word[id] for id in ids])
    
    def __len__(self):
        return len(self.word_to_id)
```

使用它：

```python
text = "Hello world. Hello NLP. NLP is fun!"
vocab = Vocabulary()
vocab.build_from_text(text)

print(vocab.word_to_id)
# 输出：
# {'<unk>': 0, 'hello': 1, 'world': 2, '.': 3, 'nlp': 4, 'is': 5, 'fun': 6, '!': 7}
```

现在，每个词都有了一个数字ID。我们可以把整段话变成一个数字序列：

```python
vocab.encode("Hello NLP!")  # 输出: [1, 4, 7]
vocab.decode([1, 3, 4])     # 输出: "hello . nlp"
```

**这一步的意义：** 把自然语言转换成了计算机能处理的**数字序列**。

## 第三步：从ID到向量——词的分布式表示（Word Embedding）

光有ID还不够。ID只是一个编号，比如“hello”是1，“nlp”是4，但1和4之间没有任何语义关系。

我们需要把每个词变成一个**向量（vector）**，也就是一串数字，比如：

```text
"hello" → [0.2, -0.5, 0.8, 1.1, -0.3]
"nlp"   → [0.1, -0.6, 0.9, 1.0, -0.2]
```

这些数字不是随机的，而是通过训练学习出来的。它们的特点是：**语义相近的词，向量也相近**。

> 注：这里需要线性代数的知识

比如，“猫”和“狗”的向量距离会很近，而“猫”和“汽车”的距离会很远。

这就是所谓的**词的分布式表示（Distributed Representation）**。

### 为什么不能用One-Hot？

你可能会想：既然每个词有ID，那直接用One-Hot编码不行吗？

比如词汇表有10万个词，那“hello”就是：

```
[0, 1, 0, 0, ..., 0]  # 长度为10万，只有一个1
```

问题来了：

- 向量太长，浪费内存
- 所有词之间都是“正交”的，无法表达相似性 (注：需要了解线性代数)
- 模型学不到“语义”

所以，我们需要更聪明的方法——**把高维稀疏的One-Hot压缩成低维密集的向量**。

这就是Word2Vec的使命。

## 第四步：Word2Vec 的核心思想——用上下文预测单词

Word2Vec 有两种模型：**CBOW** 和 **Skip-Gram**。我们先讲更直观的 **CBOW（Continuous Bag of Words）**。

### CBOW 是怎么工作的？

想象你在玩一个填空游戏：

> “You say goodbye and I say ___.”

你大概率会填“hello”。

CBOW 的任务就是：**根据上下文（context）预测中间的词（target）**。

比如窗口大小为1，那么：

| 上下文 | 目标词 |
|--------|--------|
| ['you', 'goodbye'] | 'say' |
| ['say', 'and'] | 'goodbye' |
| ['goodbye', 'I'] | 'and' |
| ['and', 'say'] | 'I' |
| ['I', 'hello'] | 'say' |

模型通过大量这样的例子学习：什么样的上下文对应什么样的词。

在这个过程中，它自动学会了每个词应该怎么用一串数字（向量）来表示，才能最好地完成预测任务。

## 第五步：构建训练数据集（Dataset）

我们写一个 `OneHotDataset` 类，自动从文本生成训练样本：

```python
class OneHotDataset(Dataset):
    def __init__(self, text: str, vocab: Vocabulary, window_size: int = 1):
        self.vocab = vocab
        self.window_size = window_size
        self.token_ids = vocab.encode(text)
        self.contexts, self.target = self.create_contexts_target()

    def create_contexts_target(self):
        target = self.token_ids[self.window_size:-self.window_size]
        contexts = []
        for idx in range(self.window_size, len(self.token_ids) - self.window_size):
            context = []
            for t in range(-self.window_size, self.window_size + 1):
                if t != 0:
                    context.append(self.token_ids[idx + t])
            contexts.append(context)
        return contexts, target

    def __getitem__(self, idx):
        ctx = self.contexts[idx]
        tgt = self.target[idx]
        return (
            F.one_hot(torch.tensor(ctx), num_classes=len(self.vocab)).float(),
            torch.tensor(tgt)
        )
```

这样，每条数据就是一个 `(上下文向量, 目标词ID)` 对。

## 第六步：搭建CBOW模型

我们用PyTorch搭建一个简单的神经网络：

```python
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, hidden_size, bias=False)  # 输入层（词向量）
        self.output_layer = nn.Linear(hidden_size, vocab_size)           # 输出层

    def forward(self, contexts):
        """
        输入: (batch_size, 2, vocab_size)
        输出: (batch_size, vocab_size)
        """
        h0 = self.embedding(contexts[:, 0])  # 第一个上下文词
        h1 = self.embedding(contexts[:, 1])  # 第二个上下文词
        h = (h0 + h1) / 2                    # 平均得到隐藏层
        out = self.output_layer(h)           # 预测目标词
        return out
```

模型结构很简单：

1. 两个上下文词 → 转为One-Hot → 通过共享的 `embedding` 层变成向量
2. 两个向量求平均 → 得到“语义总结”
3. 送入输出层 → 预测目标词的概率分布

## 第七步：训练模型

```python
text = 'You say goodbye and I say hello.'
vocab = Vocabulary()
vocab.build_from_text(text)

dataset = OneHotDataset(text, vocab)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = CBOWModel(vocab.size, hidden_size=5)

# CrossEntropyLoss 默认期望 target 是类别索引（long）。
# 默认情况下，CrossEntropyLoss 会对 batch 中每个样本的损失先求和，
# 然后根据 reduction 参数决定是否除以批量大小（mean）或保持求和（sum）。
# 默认是 reduction='mean'，也就是说：它会除以批量大小（N）。
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_losses: List[float] = []

n_epochs = 100

for epoch in range(n_epochs):
    for x, y in dataloader:
        x_batch = x.to(device)
        y_batch = y.to(device)
        
        # Step 1: 模型输出
        yhat = model(x.float())

        # Step 2: 计算损失
        loss: torch.Tensor = loss_fn(yhat, y)
        
        # Step 3: 计算梯度
        optimizer.zero_grad()
        loss.backward()

        # Step 4: Make a step
        optimizer.step()

        # .item() 是 PyTorch 张量的一个方法，专门用于：
        # 将一个只包含单个元素的张量（scalar tensor）转换为 Python 原生数值类型（如 float、int）。
        # GPU → CPU 数据传输（同步操作）
        # 如果张量在 GPU 上，PyTorch 会同步地将其从 GPU 显存拷贝到 CPU 内存。
        # 这是一个 同步点（synchronization point），会阻塞 CPU 等待 GPU 完成计算。
        # 拷贝完成后，PyTorch 提取单个值并转换为 float（如果是浮点张量）或 int（如果是整数张量）。
        batch_losses.append(loss.item())

    train_loss = sum(batch_losses) / len(batch_losses)
    print(f'Epoch {epoch+1}. Train loss: {train_loss}')
```

训练完成后，`model.embedding.weight` 就是**所有词的向量表示**！

每一列对应一个词的向量。比如第1列是“hello”的向量，第4列是“nlp”的向量。

## 结语：这就是大模型的第一步

你可能觉得这个模型很简单，训练的数据也很小，但它揭示了一个深刻的道理：

> **语言的含义，来自于它出现的上下文。**

Word2Vec 正是基于这个思想——“**分布式假设**”——让机器第一次学会了用向量表示词语的语义。

虽然今天的大型语言模型（如GPT、BERT）比Word2Vec复杂得多，但它们的第一步，依然是：

1. **分词**
2. **构建词汇表**
3. **将词转换为向量**
4. **通过上下文学习语义**

所以，Word2Vec 不是过时的技术，而是**现代NLP的奠基者**。

下一篇文章，我们将用 `nn.Embedding` 层替代手动的 `Linear` 层，进一步优化实现，并可视化词向量，看看“king - man + woman”是否真的等于“queen”。

---

**参考资料：**

- 斋藤康毅《深度学习进阶：自然语言处理》
- PyTorch官方文档


<nav class="pagination justify-content-between">
<span></span>
<a href="../">目录</a>
<a href="../2-improve-word2vec">改进Word2Vec：从“暴力计算”到“聪明学习”</a>
</nav>

