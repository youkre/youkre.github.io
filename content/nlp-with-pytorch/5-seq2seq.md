---
title: "Seq2Seq：教神经网络“中译英”——从一句话到一段话"
summary: "Seq2Seq 就像一个“双人翻译小组”，用编码器把所有意思浓缩成一个“小纸条”，再让解码器看着这张小纸条，用另一种语言说出来"
date: 2025-10-08T21:58:00+08:00
---

> 上一篇，我们教会了模型记住一句话的“上下文”。  
> 但它还不会“翻译”——比如把 “Hello” 变成 “你好”，或者把 “2+3” 算出 “5”。  
> 今天，我们就来教它一项新技能：**把一种序列，转换成另一种序列**。  
> 这就是 **Seq2Seq（Sequence to Sequence）模型**，机器翻译、聊天机器人、文本摘要的基石。

## 一句话理解 Seq2Seq

> **Seq2Seq 就像一个“双人翻译小组”**：
> **编码器（Encoder）**：是“理解专家”。它听完一整句英文，把所有意思浓缩成一个“小纸条”。
> **解码器（Decoder）**：是“表达专家”。它看着这张小纸条，用中文一句句说出来。

整个过程是：**英文句子 → 编码器 → 小纸条（上下文向量） → 解码器 → 中文句子**

## 核心组件：编码器与解码器

### 编码器（Encoder）：做个“总结党”

编码器的任务是：**把输入序列“读”懂，压缩成一个固定长度的“上下文向量”**。

它通常是一个 LSTM（或 GRU），和我们之前做的语言模型很像。

```python
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
    pad_sequence
)

class Encoder(nn.Module):
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
            # 关键：处理变长序列，跳过填充部分的计算
            packed_embeded = pack_padded_sequence(
                embedded,
                input_lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, (hn, cn) = self.lstm(packed_embeded)
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True
            )
        else:
            # output: (batch_size, seq_len, hidden_size)
            # hn: (1, batch_size, hidden_size)
            # cn: (1, batch_size, hidden_size)
            output, (hn, cn) = self.lstm(embedded)

        return output, (hn, cn) # 返回所有隐藏状态和最终状态
```

- `hn`, `cn`：就是那个“小纸条”，包含了输入序列的全部信息。
- `pack_padded_sequence` 和 `pad_packed_sequence`：这是处理**变长句子**的关键！  
  
想象班级里有高有矮的学生，我们按身高排队，把短句子“卷起来”不计算，避免“填充”的0干扰记忆。

### 解码器（Decoder）：做个“复述者”

解码器的任务是：**拿着“小纸条”，生成目标序列**。

它也是一个 LSTM，但启动方式不同：

- **初始状态**：用编码器的 `hn`, `cn` 来初始化。
- **输入**：在训练时，用“**教师强制（Teacher Forcing）**”——直接把正确答案喂给它，让它学得更快。

```python
class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.affine = nn.Linear(hidden_size, vocab_size)

    def forward(self, xs: torch.Tensor, h_c: Tuple[torch.Tensor, torch.Tensor], input_lengths: Optional[torch.Tensor] = None):
        """
        xs: (batch_size, seq_len)
        h_c: tuple of (h_0, c_0)
            h_0: (1, batch_size, hidden_size)
            c_0: (1, batch_size, hidden_size)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
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
            packed_output, (hn, cn) = self.lstm(packed_embeded, h_c) # 用编码器的状态初始化
            xs, _ = pad_packed_sequence(
                packed_output,
                batch_first=True
            )
        else:
            # xs: (batch_size, seq_len, hidden_size)
            # hn: (1, batch_size, hidden_size)
            # cn: (1, batch_size, hidden_size)
            xs, (hn, cn) = self.lstm(xs, h_c)
        # (batch_size, seq_len, vocab_size)
        logits = self.affine(xs)
        return logits, (hn, cn)
    
    def generate(
            self, 
            h_c: Tuple[torch.Tensor, torch.Tensor], 
            start_id: int, 
            sample_size: int,
            end_id: Optional[int] = None,
        ):
        """
        生成文本
        h_c: 初始隐藏状态 (h_0, c_0)，每个形状为 (1, 1, hidden_size)
        start_id: 起始 token ID
        sample_size: 生成多少个词
        """
        sampled: List[int] = []
        x = torch.tensor([[start_id]]) # (1, 1)
        h, c = h_c
        sample_id = start_id

        for _ in range(sample_size):
            if end_id is not None and sample_id == end_id:
                break
            
            out = self.embedding(x) # (1, 1, D)
            out, (h, c) = self.lstm(out, (h, c)) # 更新 h, c
            score: torch.Tensor = self.affine(out) # (1, 1, V)
            sample_id = score.argmax(dim=-1).item() # 取最大概率的词
            sampled.append(int(sample_id))
            x = torch.tensor([[sample_id]]) # 用于下一次输入

        return sampled
```

> **关键点**：解码器是“自回归”的——它生成的每个词，都可能成为下一个词的输入。

## 完整模型：把两人组合起来

现在，我们把编码器和解码器组装成一个完整的 Seq2Seq 模型：

```python
class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int):
        super().__init__()
        self.encoder = Encoder(vocab_size, embedding_dim, hidden_size)
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size)
    
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
        encoder_output, (hn, cn) = self.encoder(enc_input, enc_lens)
        # logits: (batch_size, seq_len, vocab_size)
        logits, _ = self.decoder(dec_input, (hn, cn), dec_lens)
        return logits
    
    def generate(self, x: torch.Tensor, start_id: int, sample_size: int):
        """
        生成序列
        enc_input: (1, enc_seq_len) 或 (B, enc_seq_len)
        """
        _, (hn, cn) = self.encoder(x)
        sampled = self.decoder.generate((hn, cn), start_id, sample_size)
        return sampled
```

训练时，我们喂给它：

- `enc_input`：输入序列（如 `"2+3"`）
- `dec_input`：目标序列（如 `"<start>5<end>"`，加上起始和结束标记）
- `enc_lens`, `dec_lens`：输入序列的实际长度（用于打包）

## 数据准备：加法数据集实战

我们用一个简单的“加法数据集”来训练模型，让它学会做算术。

数据长这样：

```
2+3_5
12+8_20
...
```

### 1. 构建词汇表

```python
root_dir = Path.home() / 'datasets/dl-nlp'
addition_file = root_dir / 'addition.txt'
out_dir = root_dir / 'addition'
out_dir.mkdir(parents=True, exist_ok=True)

def load_data(filepath: Path) -> List[Tuple[str, str]]:
    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} does not exist")
    
    pairs: List[Tuple[str, str]] = []
    for line in filepath.open(encoding="utf-8"):
        src, tgt = line.strip().split("_")
        pairs.append((src, tgt))
    
    return pairs

def split_data(data: List[Tuple[str, str]]):
    split_at = len(data) - len(data) // 10
    train, test = data[:split_at], data[split_at:]

    return train, test

class Seq2SeqVocabulary:
    def __init__(
            self,
        ):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self._add_special_tokens()

    @property
    def pad_id(self):
        return self.word_to_id[self.PAT_TOKEN]
    
    @property
    def unk_id(self):
        return self.word_to_id[self.UNK_TOKEN]
    
    @property
    def start_id(self):
        return self.word_to_id[self.START_TOKEN]
    
    def _add_special_tokens(self):
        self.PAT_TOKEN= '<pad>'
        self.START_TOKEN = '<start>'
        self.END_TOKEN = '<end>'
        self.UNK_TOKEN = '<unk>'

        for token in [self.PAT_TOKEN, self.START_TOKEN, self.END_TOKEN, self.UNK_TOKEN]:
            self._add(token)
    
    def _add(self, word: str):
        if word not in self.word_to_id:
            idx = len(self.word_to_id)
            self.word_to_id[word] = idx
            self.id_to_word[idx] = word

    
    
    def build(self, sentences: List[str]):
        chars = set()
        for sent in sentences:
            # str 被当作字符序列
            chars.update(sent)

        for char in chars:
            self._add(char)

    def build_from_pairs(self, pairs: List[Tuple[str, str]]):
        for src, tgt in pairs:
            self.build([src, tgt])

    def encode(self, text: str) -> List[int]:
        return [
            self.word_to_id.get(c, self.unk_id) 
            for c in text
        ]
    
    def decode(self, ids: List[int]) -> str:
        words = []
        for idx in ids:
            word = self.id_to_word.get(idx, self.UNK_TOKEN)
            if word == self.END_TOKEN:
                break
            if word not in [self.PAT_TOKEN, self.START_TOKEN]:
                words.append(word)
        
        return ''.join(words)
    
    def save(self, path: Path):
        vocab = {
            'word_to_id': self.word_to_id,
        }
        path.mkdir(parents=True, exist_ok=True)
        data = json.dumps(vocab, ensure_ascii=False, indent=4)
        filepath = path.joinpath('vocab.json')
        filepath.write_text(data)
        print(f'Vocabulary saved {filepath}')

    def load(self, path: Path):
        with path.joinpath('vocab.json').open('r', encoding='utf-8') as f:
            vocab = json.load(f)

        self.word_to_id = vocab['word_to_id']
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}
    
    @property
    def size(self):
        return len(self.word_to_id)
    
    def __len__(self):
        return self.size

class AdditionDataset(Dataset):
    def __init__(
        self,
        vocab: Seq2SeqVocabulary,
        corpus: List[Tuple[str, str]],
    ):
        self.vocab = vocab
        self.corpus = corpus

    def __len__(self):
        return len(self.corpus)

    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        src_sent, tgt_sent = self.corpus[idx]
        
        # Encoder 输入: "3+2" -> [3, +, 2]
        encoder_input = self.vocab.encode(src_sent)
        
        # Decoder 输入: "<start>5" -> [<start>, 5]
        decoder_input = [
            self.vocab.word_to_id[self.vocab.START_TOKEN]
        ] + self.vocab.encode(tgt_sent)

        # Loss 目标: "5<end>" -> [5, <end>]
        decoder_output = self.vocab.encode(tgt_sent) + [
            self.vocab.word_to_id[self.vocab.END_TOKEN]
        ]

        return {
            'encoder_input': torch.tensor(encoder_input, dtype=torch.long),
            'decoder_input': torch.tensor(decoder_input, dtype=torch.long),
            'decoder_output': torch.tensor(decoder_output, dtype=torch.long)
        }
```

我们按**字符**级别切分，所以词汇表里是 `'0'~'9'`, `'+'`, `'-'` 等单个字符。

为了让模型拿到形状相同的张量，需要在 collate 函数中填充序列长度为最长的序列，还要返回每个序列的实际长度：

```python
def collate_fn(batch, padding_value: int):
    encoder_inputs = [b['encoder_input'] for b in batch]
    decoder_inputs = [b['decoder_input'] for b in batch]
    decoder_outputs = [b['decoder_output'] for b in batch]

    # 获取每个序列的实际长度
    encoder_input_lengths = torch.tensor([len(seq) for seq in encoder_inputs])
    decoder_input_lengths = torch.tensor([len(seq) for seq in decoder_inputs])  # ← 注意：decoder 输入长度


    encoder_inputs = pad_sequence(
        encoder_inputs, 
        batch_first=True, 
        padding_value=padding_value
    )
    decoder_inputs = pad_sequence(
        decoder_inputs,
        batch_first=True,
        padding_value=padding_value
    )
    decoder_outputs = pad_sequence(
        decoder_outputs,
        batch_first=True,
        padding_value=padding_value
    )
    
    return {
        'encoder_input': encoder_inputs,
        'encoder_input_lengths': encoder_input_lengths,  # 添加序列长度
        'decoder_input': decoder_inputs,
        'decoder_input_lengths': decoder_input_lengths,
        'decoder_output': decoder_outputs,
    }
```

### 2. 训练

准备数据：

```python
add_data = load_data(addition_file)
train, test = split_data(add_data)
vocab = Seq2SeqVocabulary()
vocab.build_from_pairs(add_data)

dataset = AdditionDataset(vocab, train)

dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda batch: collate_fn(batch, vocab.pad_id)
)

vocab_size = vocab.size
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 50
lr = 01e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = Seq2Seq(
    vocab_size=vocab_size,
    embedding_dim=wordvec_size,
    hidden_size=hidden_size
)
optimizer = optim.Adam(model.parameters(), lr=lr)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

model.to(device)
```

训练循环：

```python
for epoch in range(max_epoch):
    batch_losses: List[float] = []
    model.train()
    for batch in dataloader:
        enc_input = batch['encoder_input'].to(device) # (B, T_enc)
        enc_len = batch['encoder_input_lengths']    # (B,)
        dec_input = batch['decoder_input'].to(device) # (B, T_dec)
        dec_len = batch['decoder_input_lengths']
        dec_output = batch['decoder_output'].to(device) # (B, T_dec)

        # Step 1: Compute the output
        logits = model(enc_input, dec_input, enc_len, dec_len)
        # Step 2: Compute the loss
        loss: torch.Tensor = loss_fn(logits.flatten(0, 1), dec_output.flatten())
        # Step 3: Compute gradients
        optimizer.zero_grad()
        loss.backward()
        # Step 4: Make a step
        optimizer.step()

        batch_losses.append(loss.item())

    print(f'Epoch {epoch}: {sum(batch_losses) / len(batch_losses)}')
```

> **`ignore_index=vocab.pad_id`**：告诉损失函数，“`<pad>` 是占位符，别算它的损失”，非常方便。

## 总结：Seq2Seq 的智慧

我们学会了：

1. **Seq2Seq = 编码器 + 解码器**：一个理解，一个生成。
2. **上下文向量是“小纸条”**：承载输入的全部信息。
3. **教师强制加速训练**：用正确答案引导解码器。
4. **`pack_padded_sequence` 处理变长序列**：跳过填充，高效计算。
5. **`CrossEntropyLoss(ignore_index=...)` 忽略填充**：让损失计算更智能。

---

**参考资料：**

- 斋藤康毅《深度学习进阶：自然语言处理》
- PyTorch官方文档


<nav class="pagination justify-content-between">
<a href="../4-lstm">LSTM：给神经网络装上“长期记忆”</a>
<a href="../">目录</a>
<a href="../6-attention">注意力机制：让神经网络学会“重点回顾”</a>
</nav>

