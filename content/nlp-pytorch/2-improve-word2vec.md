---
title: 改进Word2Vec：从“暴力计算”到“聪明学习”
description: Word2Vec 原始模型计算太慢？本文带你升级！用 nn.Embedding 替代 
  one-hot，高效提取词向量。引入负采样，化“大海捞针”为“真假判断”，大幅加速训练。代码实战，教你打造聪明高效的词向量模型。
draft: false
isCJKLanguage: true
keywords: []
summary: Word2Vec 原始模型计算太慢？本文带你升级！用 nn.Embedding 替代 
  one-hot，高效提取词向量。引入负采样，化“大海捞针”为“真假判断”，大幅加速训练。代码实战，教你打造聪明高效的词向量模型。
date: '2025-10-07'
---

> 上一篇，我们用一个“填空游戏”教会了模型预测单词——这就是经典的 **CBOW 模型**。<br>
> 但我们的模型有个“小毛病”：它太“笨”了，每次都要把整个词表过一遍，像个学生每次考试都要把字典从头到尾背一遍。<br>
> 今天，我们就来给它“升级装备”，让它变得更聪明、更高效！

## 问题来了：模型太“累”了

还记得上一篇的模型是怎么工作的吗？

1. 输入一个上下文（比如 `["you", "goodbye"]`）。
2. 模型要计算这个词组合和**词表中每一个词**的匹配程度。
3. 最后输出一个长长的概率列表，告诉我们下一个词最可能是“say”、“hello”还是“cat”。

这听起来没问题，但如果词表有 **10万个词** 呢？模型每次都要做一次 300×100,000 的矩阵乘法（假设向量是300维），这就像让一个学生每次只猜一个字，却要写完10万道选择题！

更糟糕的是，训练时反向传播要更新 **3000万个参数**（300 × 100,000），这简直是“灾难级”的计算开销。

> **一句话总结问题**：我们让模型做了一道“10万选1”的超难选择题，它累垮了！

## 升级第一步：告别One-Hot，拥抱 `nn.Embedding`

还记得我们是怎么把“hello”变成向量的吗？先变成 one-hot 向量 `[0,1,0,0,...]`，再乘以一个大矩阵。

这就像：你要从10万个抽屉里拿一个文件，先写一张“我要第2号抽屉”的纸条（one-hot），然后让机器人把这张纸条和10万个抽屉的标签一一比对，最后才打开第2号抽屉。

太麻烦了！为什么不直接告诉机器人：“去2号抽屉拿文件”？

这就是 `nn.Embedding` 层的智慧！

### `nn.Embedding`：你的“智能抽屉管理员”

```python
self.embedding = nn.Embedding(vocab_size, hidden_size)
```

- `vocab_size=100000`：总共有10万个抽屉。
- `hidden_size=300`：每个抽屉里放一个300维的向量。

**它怎么工作？**

直接给它一个词的ID（比如 `2`），它瞬间就返回对应的词向量，就像机器人直接去2号抽屉取文件，跳过了繁琐的比对过程。

> **好处**：计算快、内存省、代码简洁。

我们把上一篇的 `nn.Linear` 换成 `nn.Embedding`，模型就轻装上阵了：

```python
class CBOWModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()
        # 替换掉 Linear
        # Embedding 层的输入(*) 是任意形状的 IntTensor 或 LongTensor，这是要提取的索引
        # 输出(*, H) 其中 * 是输入形状
        self.embedding = nn.Embedding(vocab_size, hidden_size)  # 输入层 -> 隐藏层
        self.output_layer = nn.Linear(hidden_size, vocab_size)  # 隐藏层 -> 输出层

    def forward(self, contexts: torch.Tensor):
        """
        输入: (batch_size, 2)
        输出: (batch_size, vocab_size)
        """
        # (batch_size, hidden_size)
        h0 = self.embedding(contexts[:, 0]) # 直接索引，快！
        # (batch_size, hidden_size)
        h1 = self.embedding(contexts[:, 1])
        # (batch_size, hidden_size)
        h = (h0 + h1) * 0.5
        # (batch_size, vocab_size)
        out = self.output_layer(h)
        return out
```

**但问题还没解决！** 虽然取向量快了，但最后的 `output_layer` 还是要对10万个词做计算。模型还是太累。

## 升级第二步：负采样——“化整为零”的聪明学习法

我们得换个思路：**不搞“10万选1”的大考，改搞“1对多”的小测验。**

这就是 **负采样（Negative Sampling）** 的核心思想。

### 负采样：老师出题的新方式

想象你是老师，想考学生对“say”这个词的掌握。

**老方法（效率低）：**

> “you say goodbye and I ___ hello” 的选项是：
> A. say  B. hello  C. goodbye  D. and  E. I  ...（共10万个选项）
> 学生要看完所有选项才能选B。

**新方法（负采样，效率高）：**

> “you say goodbye and I ___ hello” 的选项是：
> A. say（正确答案）<br>
> B. cat（乱选一个错的）<br>
> C. computer（再乱选一个）<br>
> D. love（再选一个）<br>
> E. run（再选一个）

只需要判断：“A是对的，B、C、D、E都是错的”。这题简单多了！

- **正样本（Positive Sample）**：真实的上下文+目标词组合，比如 `(["you", "goodbye"], "say")`。
- **负样本（Negative Sample）**：同一个上下文 + 一个**随机选的、错误的词**，比如 `(["you", "goodbye"], "cat")`。

我们的目标变成了：

- 让模型认为“正样本”是真的（得分接近1）。
- 让模型认为“负样本”是假的（得分接近0）。

这不再是一个“多分类”问题，而是一个“二分类”问题，但每个 batch 里有多个二分类任务（1个正样本 + K个负样本）。

### 实现负采样

我们先写一个函数，给每个目标词采样出几个“替身”（负样本）：

```python
def sample_negative_words(target_ids: torch.Tensor, vocab_size: int, num_neg_samples: int):
    """
    从词汇表中采样出不包含目标词的负样本。
    target_ids: (batch_size,)
    输出: (batch_size, num_neg_samples)
    num_neg_samples 每个元素是一个负采样单词的索引
    """
    batch_size = target_ids.size(0)

    # # 每个词初始权重都是1 (batch_size, vocab_size)
    weights = torch.ones(batch_size, vocab_size)

    # # 把目标词的权重设为0，让它不会被选中
    weights[torch.arange(batch_size), target_ids] = 0

    # 使用 multinomial 进行采样（不放回）
    # 从每一行中选择 num_neg_samples 个元素的索引，由于目标词的位置为0，所以会跳过目标词。
    # (batch_zie, num_neg_samples)
    negative_word_ids = torch.multinomial(weights, num_neg_samples, replacement=False)

    return negative_word_ids
```

就像抽奖时，把正确答案的抽奖券撕掉，确保不会抽到自己。

### 搭建新模型

现在我们设计一个支持负采样的CBOW模型：

```python
class CBOWNegativeSampling(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()
        # 两个嵌入层：一个给上下文词用，一个给目标词用
        self.context_embedding = nn.Embedding(vocab_size, embedding_dim)  # 上下文向量
        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)     # 目标词向量

    def forward(self, context_ids: torch.Tensor, input_ids: torch.Tensor):
        """
        context_ids: (batch_size, context_size) 上下文词ID
        input_ids:   (batch_size, 1 + K)        目标词ID + K个负样本ID
        """
        # 1. 获取上下文向量并平均
        # (batch_size, context_size, embedding_size)
        context_vecs = self.context_embedding(context_ids)
        # (batch_size, embedding_size)
        context_mean = torch.mean(context_vecs, dim=1)

        # 2. 获取目标词和负样本的向量
        # (batch_size, 1+K, embedding_dim)
        word_vecs = self.word_embedding(input_ids)

        # 3. 计算点积得分（相似度）
        # bmm: 把每个上下文平均向量和每个候选词向量做点积
        # context_mean.unsqueeze(2) 添加一个维度：
        # (batch_size, embedding_size) -> (batch_size, embedding_dim, 1)
        # torch.bmm 操作：
        # (batch_size, 1+K, embedding_dim) @ (batch_size, embedding_dim, 1)
        # -> (batch_size, 1+K, 1)
        scores = torch.bmm(word_vecs, context_mean.unsqueeze(2))
        # squeeze(2) -> (batch_size, 1+K)
        scores = scores.squeeze(2)

        return scores  # 返回原始得分，loss会自动加sigmoid
```

> 🔍 **为什么有两个嵌入层？**
> 理论上可以共享，但实践中常分开。你可以理解为“上下文角色”和“目标词角色”略有不同，分开学习更灵活。

> 🔍 **为什么返回 `scores` 而不是 `sigmoid(scores)`？**
> 因为我们会用 `BCEWithLogitsLoss`，它内部会做 `sigmoid`，这样数值更稳定。

### 损失函数：二分类交叉熵

```python
criterion = nn.BCEWithLogitsLoss()  # 自带 sigmoid，更稳定
```

标签长这样：

```python
labels = torch.zeros(batch_size, 1 + num_neg_samples)
labels[:, 0] = 1.0  # 只有第一个（正样本）是1，其余都是0
```

## 数据流水线：`collate_fn` 的妙用

用 Dateset 读取原始数据提供给 DataLoader：

```python
# 这个类负责从原始语料中提取上下文和目标词对。
class CBOWDataset(Dataset):
    def __init__(self, corpus: List[int], window_size: int = 1):
        self.corpus = corpus # 已经转换为 token ID 的列表
        self.window_size = window_size
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < self.window_size or idx >= len(self.corpus) - self.window_size:
            return {
                'context_ids': torch.tensor([], dtype=torch.long),
                'target_id': torch.tensor(-1, dtype=torch.long)
            }
        target_id = self.corpus[idx]
        context_ids: List[int] = []
        for i in range(-self.window_size, self.window_size + 1):
            if i == 0:
                continue
            context_ids.append(self.corpus[idx + i])

        return {
            'context_ids': torch.tensor(context_ids, dtype=torch.long), # shape: (window_size * 2, )
            'target_id': torch.tensor(target_id, dtype=torch.long) # shape: (1,)
        }
```

负采样需要在每个 batch 生成，但我们不想把这种“训练逻辑”塞进 `Dataset`。怎么办？

PyTorch 提供了 `collate_fn`——它像一个“打包助手”，在数据送入模型前，把零散的样本打包成 batch，并做额外处理。

```python
def collate_fn(batch: List[Dict[str, torch.Tensor]], num_neg_samples: int, vocab_size: int):
    # 过滤无效样本
    batch = [item for item in batch if item['context_ids'].numel() > 0]
    
    # 提取上下文和目标词
    # (batch_size, context_size)
    # 例如 b['context_ids'] 是 [6, 2]
    # Stack到 [1, 2] 得到
    # [[6, 2],
    #  [1, 2]]
    context_ids = torch.stack([b['context_ids'] for b in batch])
    # 把 只有一个值的 tensor 构成的数组转换成 tensor
    # [torch.tensor(1), torch.tensor(1)] -> torch.tensor([1, 1])
    # (batch_size)
    target_ids = torch.stack([b['target_id'] for b in batch])
    
    # 生成负样本
    # (batch_size, num_neg_samples)
    neg_ids = sample_negative_words(target_ids, vocab_size, num_neg_samples)
    
    # 拼接正负样本：[正, 负1, 负2, ...]
    # target_ids.unsqueeze(1) -> (batch_size, 1)
    # （batch_size, 1 + K)
    input_ids = torch.cat([target_ids.unsqueeze(1), neg_ids], dim=1)
    
    # 生成标签
    # 第一列是正样本 (1)，其余是负样本 (0)
    labels = torch.zeros_like(input_ids, dtype=torch.float)
    labels[:, 0] = 1.0
    
    return {
        'context_ids': context_ids, # shape: (batch_size, 2*window_size)
        'input_ids': input_ids, # shape: (batch_size, 1 + K)
        'labels': labels # shape: (batch_size, 1 + K)
    }
```

这样，`Dataset` 只负责提供原始样本，`collate_fn` 负责“加工”成训练所需格式，职责分明！

## 实战：在PTB数据集上训练

最后，我们用真实数据训练这个升级版模型。

PTB 数据集下载地址：

- https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.train.txt
- https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.valid.txt
- https://raw.githubusercontent.com/tomsercu/lstm/master/data/ptb.test.txt

### 步骤 1：Vocabulary

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

    def add_special_token(self, token: str):
        if token not in self.word_to_id:
            new_id = len(self.word_to_id)
            self.word_to_id[token] = new_id
            self.id_to_word[new_id] = token
            self.word_freq[token] = 0
        
    def tokenize(self, text: str):
        return self.tokenizer(text)
    
    def build(self, text: str, min_freq: int = 1):
        for word in self.tokenize(text):
            self.word_freq[word] += 1
            if self.word_freq[word] >= min_freq and word not in self.word_to_id:
                new_id = len(self.word_to_id)
                self.word_to_id[word] = new_id
                self.id_to_word[new_id] = word

    def build_from_sentences(self, sentences: List[str], min_freq: int = 1):
        for sentence in sentences:
            self.build(sentence, min_freq)

    def encode(self, text: str) -> List[int]:
        return [self.word_to_id.get(word, 0) for word in self.tokenize(text)]
    
    def decode(self, ids: List[int]) -> str:
        return ' '.join([self.id_to_word[id] for id in ids])
    
    @property
    def size(self):
        return len(self.word_to_id)
    
    def __len__(self):
        return self.size
```

### 步骤 2：PTBCbowDataset

这个类负责：

- 加载文本
- 使用 vocab 转为 ID
- 为每个中心词生成 (context, target) 样本

```python
class PTBCBOWDataset(Dataset):
    def __init__(
        self, 
        file_path: Path,
        vocab: Vocabulary,
        window_size: int = 2,
    ):
        self.file_path = file_path
        self.window_size = window_size
        self.sentences = self._load_sentences()
        
        vocab.build_from_sentences(self.sentences)
        self.vocab = vocab

        self.sentences_ids = [self.vocab.encode(sent) for sent in self.sentences]
        self.corpus = self.create_contexts_target()


    def _load_sentences(self) -> List[str]:
        with open(self.file_path, 'r', encoding='utf-8') as f:
            sents = [line.strip() for line in f if line.strip()]
            return sents
        
    def create_contexts_target(self) -> List[Dict]:
        samples = []
        for sent_ids in self.sentences_ids:
            for idx in range(self.window_size, len(sent_ids) - self.window_size):
                context_ids = sent_ids[idx - self.window_size: idx] + sent_ids[idx + 1: idx + self.window_size + 1]
                target_id = sent_ids[idx]
                samples.append({
                    'context_ids': torch.tensor(context_ids, dtype=torch.long),
                    'target_id': torch.tensor(target_id, dtype=torch.long),
                })

        return samples
        
    
    def __len__(self):
        return len(self.corpus)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.corpus[idx]
```

### 步骤 3：创建数据集和数据加载器

```python
data_dir = Path.home() / 'datasets' / 'ptb'
train_file = data_dir / 'ptb.train.txt'

# 1. 加载数据和构建词汇表
vocab = Vocabulary()
train_dataset = PTBCBOWDataset(
    train_file,
    vocab=vocab, 
    window_size=2
)
print("词汇表大小:", len(vocab))

# 2. 创建DataLoader，使用collate_fn
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=lambda b: collate_fn(b, num_neg_samples=5, vocab_size=len(vocab))
)

# 3. 初始化模型
model = CBOWNegativeSampling(vocab_size=len(vocab), embedding_dim=100)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.BCEWithLogitsLoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)
```

### 步骤 4：训练模型

```python
for epoch in range(10):
    losses = []
    model.train()
    for batch in train_loader:
        ctx = batch['context_ids'].to(device)
        inp = batch['input_ids'].to(device)
        lbl = batch['labels'].to(device)

        logits = model(ctx, inp)
        loss = criterion(logits, lbl)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    print(f"Epoch {epoch}, 平均损失: {sum(losses)/len(losses):.4f}")
```

## 总结：我们做了什么？

我们给Word2Vec模型做了两次关键升级：

1. **`nn.Embedding` → 高效提取词向量** 告别了笨重的 one-hot 和矩阵乘法，像查字典一样快速获取词向量。

2. **负采样 → 化繁为简的训练策略** 把“大海捞针”式的多分类，变成了“真假判断”式的二分类，大大降低了计算负担。

---

**参考资料：**

- 斋藤康毅《深度学习进阶：自然语言处理》
- PyTorch官方文档