import os
import sys
import time
import re
import random
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

# =========================================================================
# 1. 参数配置 (Argument Parsing)
# =========================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Text Classification with RNSA Optimizer")

    # 数据参数
    parser.add_argument('--data_file', type=str, default='.\data\data.csv', help='Data file path')
    parser.add_argument('--vocab_min_freq', type=int, default=2, help='Minimal word frequency')
    parser.add_argument('--max_length', type=int, default=0, help='Max sequence length (0 for auto)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # 模型参数
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'cnn'], help='Model type')
    parser.add_argument('--embed_dim', type=int, default=100, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden size for LSTM')
    parser.add_argument('--filter_sizes', type=str, default='3,4,5', help='CNN filter sizes')
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters per size for CNN')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')

    # 训练参数
    parser.add_argument('--optimizer', type=str, default='rnsa', choices=['gd', 'momentum', 'adam', 'rnsa'], help='Optimizer')
    parser.add_argument('--lr', type=float, default=0.5, help='Learning rate (For SGD/RNSA)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--grad_threshold', type=float, default=2.0, help='Gradient clipping threshold')
    
    # 工程化参数
    parser.add_argument('--eval_every', type=int, default=500, help='Evaluate every X steps')
    parser.add_argument('--save_dir', type=str, default='runs', help='Directory to save logs and models')

    return parser.parse_args()

args = parse_args()

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 日志目录
timestamp = str(int(time.time()))
out_dir = os.path.join(args.save_dir, f"{args.model_type}_{args.optimizer}_{timestamp}")
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
writer = SummaryWriter(log_dir=out_dir)

# =========================================================================
# 2. 数据处理 (修复了 Header 问题和 Stratify 问题)
# =========================================================================
def load_or_generate_data(filename):
    if not os.path.isfile(filename):
        print(f"File not found: {filename}. Generating synthetic data...")
        # ... (模拟数据生成逻辑保持不变) ...
        num_sim = 2000
        vocab_list = ["good", "bad", "happy", "sad", "neutral", "excellent", "terrible", "okay", "great", "awful"]
        data, labels = [], []
        for _ in range(num_sim):
            length = random.randint(5, 15)
            words = [random.choice(vocab_list) for _ in range(length)]
            text = " ".join(words)
            pos_cnt = sum(1 for w in words if w in ["good", "happy", "excellent", "great"])
            neg_cnt = sum(1 for w in words if w in ["bad", "sad", "terrible", "awful"])
            label = "Positive" if pos_cnt >= neg_cnt else "Negative"
            data.append(text)
            labels.append(label)
        return pd.DataFrame({'Var1': labels, 'Var2': data})
    else:
        print(f"Reading file: {filename} ...")
        # 【修复1】尝试自动检测 Header，避免把第一行当数据
        try:
            # 先试着读一行
            df_sample = pd.read_csv(filename, nrows=5)
            # 如果第一列看起来像文本而不是标签（比如包含 'label' 字样），可能是有 Header
            # 这里我们采用一个简单的策略：如果用户没给 header 且第一行是 unique 的字符串，可能就是 header
            # 为了保险，我们直接用 header=None 读取，然后手动清洗
            # df = pd.read_csv(filename, header=None, encoding='utf-8', on_bad_lines='skip')
            # 兼容老版本 pandas: 使用 error_bad_lines=False 替代 on_bad_lines='skip'
            df = pd.read_csv(filename, header=None, encoding='utf-8', error_bad_lines=False)
        except Exception as e:
            print(f"Error reading CSV: {e}")
            sys.exit(1)
        
        # 将所有列转为字符串，防止数值型报错
        df = df.astype(str)
        
        # 【修复2】假设第一列是标签，第二列是文本
        # 检查是否包含 Header 行 (例如 "label", "text" 或 "type", "content")
        first_row = df.iloc[0].str.lower().tolist()
        potential_headers = ['label', 'type', 'class', 'target', 'text', 'content', 'review', 'var1', 'var2']
        if any(h in first_row for h in potential_headers):
            print("Detected potential header row. Removing it.")
            df = df.iloc[1:]
            
        # 重命名列以便后续处理
        df.columns = ['Var1', 'Var2'] + list(df.columns[2:]) # 只要前两列
        return df[['Var1', 'Var2']]

def tokenizer(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

def build_vocab(texts, min_freq=2):
    counter = Counter()
    for text in texts:
        counter.update(tokenizer(text))
    vocab = {'<pad>': 0, '<unk>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

def text_to_sequence(text, vocab, target_len):
    tokens = tokenizer(text)
    seq = [vocab.get(token, vocab['<unk>']) for token in tokens]
    if len(seq) < target_len:
        seq = seq + [vocab['<pad>']] * (target_len - len(seq))
    else:
        seq = seq[:target_len]
    return seq

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, seq_len, label_map):
        self.data = [text_to_sequence(t, vocab, seq_len) for t in texts]
        self.labels = [label_map[l] for l in labels]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# --- 执行数据加载 ---
df = load_or_generate_data(args.data_file)
all_texts = df['Var2'].tolist()
all_labels = df['Var1'].tolist()

# 打印标签分布，用于调试
print("Label Distribution:", Counter(all_labels))

# 【修复3】移除样本数极少的类别
label_counts = Counter(all_labels)
valid_labels = [label for label, count in label_counts.items() if count > 1]
if len(valid_labels) < len(label_counts):
    print(f"Warning: Removed classes with only 1 sample: {set(label_counts.keys()) - set(valid_labels)}")
    # 过滤 DataFrame
    df = df[df['Var1'].isin(valid_labels)]
    all_texts = df['Var2'].tolist()
    all_labels = df['Var1'].tolist()

vocab = build_vocab(all_texts, args.vocab_min_freq)
if args.max_length == 0:
    seq_lens = [len(tokenizer(t)) for t in all_texts]
    if len(seq_lens) > 0:
        args.max_length = int(np.percentile(seq_lens, 95))
    else:
        args.max_length = 50 # Fallback
print(f"Vocab Size: {len(vocab)}, Max Length: {args.max_length}")

label_map = {l: i for i, l in enumerate(sorted(set(all_labels)))}
num_classes = len(label_map)

# 【修复4】根据类别数量决定是否使用 Stratify
# 即使过滤了只有1个样本的类，train_test_split 仍然要求 test_size 对应的样本数 >=1
# 为了安全，如果总样本量太少，或者某些类太少，关闭 stratify
min_class_samples = min(Counter(all_labels).values())
if min_class_samples < 5:
    print("Warning: Some classes have very few samples. Disabling stratification.")
    stratify_option = None
else:
    stratify_option = all_labels

X_train, X_test, y_train, y_test = train_test_split(
    all_texts, all_labels, 
    test_size=0.1, 
    stratify=stratify_option, 
    random_state=args.seed
)

train_dataset = TextDataset(X_train, y_train, vocab, args.max_length, label_map)
test_dataset = TextDataset(X_test, y_test, vocab, args.max_length, label_map)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# =========================================================================
# 3. 模型定义 (LSTM & TextCNN)
# =========================================================================
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        embedded = self.dropout(self.embedding(x))
        output, (h_n, c_n) = self.lstm(embedded)
        hidden = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1)
        return self.fc(hidden)

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, filter_sizes, num_filters, output_dim, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.permute(0, 2, 1) # [B, Emb, Seq]
        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # MaxPool over time
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = self.dropout(torch.cat(pooled, dim=1))
        return self.fc(cat)

def get_model():
    if args.model_type == 'lstm':
        model = TextLSTM(len(vocab), args.embed_dim, args.hidden_dim, num_classes, args.dropout)
    elif args.model_type == 'cnn':
        filter_sizes = [int(k) for k in args.filter_sizes.split(',')]
        model = TextCNN(len(vocab), args.embed_dim, filter_sizes, args.num_filters, num_classes, args.dropout)
    
    model.to(device)
    # 初始化
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    return model

# =========================================================================
# 4. 辅助函数 & RNSA 核心
# =========================================================================
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def clip_gradients(model, threshold):
    torch.nn.utils.clip_grad_norm_(model.parameters(), threshold)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return total_loss / (total + 1e-10), correct / (total + 1e-10)

# =========================================================================
# 5. 训练循环 (RNSA Observer)
# =========================================================================
def train():
    model = get_model()
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif args.optimizer in ['gd', 'rnsa']:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'momentum':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    
    rnsa_active = (args.optimizer == 'rnsa')
    refined_model = copy.deepcopy(model) if rnsa_active else None
    grad_norm_prev = 0.0
    
    global_step = 0
    best_acc = 0.0
    
    print(f"\nStart Training: {args.model_type.upper()} | Opt: {args.optimizer.upper()} | LR: {args.lr}")
    
    for epoch in range(args.num_epochs):
        model.train()
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            global_step += 1
            
            # --- Base GD Step ---
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            
            clip_gradients(model, args.grad_threshold)
            
            # RNSA: Capture old params
            if rnsa_active:
                grad_norm_curr = get_grad_norm(model)
                old_params = {name: p.data.clone() for name, p in model.named_parameters()}
            
            # Update
            optimizer.step()
            
            # --- RNSA Logic ---
            rnsa_loss_val = None
            if rnsa_active and global_step > 1 and grad_norm_prev > 1e-10:
                w_k = grad_norm_curr / grad_norm_prev
                
                # Eta Calculation
                if abs(1 - w_k) < 1e-3: eta_k = 0.0
                else: eta_k = w_k / (1 - w_k)
                if abs(eta_k) > 5.0: eta_k = 0.0
                
                # Apply Extrapolation to Refined Model
                with torch.no_grad():
                    refined_model.eval()
                    for name, p_ref in refined_model.named_parameters():
                        p_new = dict(model.named_parameters())[name].data
                        p_old = old_params[name]
                        p_ref.data.copy_(p_new + eta_k * (p_new - p_old))
                    
                    # Compute Virtual Loss
                    out_ref = refined_model(inputs)
                    rnsa_loss_val = criterion(out_ref, targets).item()
                
                writer.add_scalar('RNSA/eta', eta_k, global_step)
                writer.add_scalar('RNSA/w_k', w_k, global_step)
            
            if rnsa_active:
                grad_norm_prev = grad_norm_curr

            # Logging
            if global_step % 10 == 0:
                writer.add_scalar('Train/Loss_Base', loss.item(), global_step)
                if rnsa_loss_val:
                    writer.add_scalar('Train/Loss_Refined', rnsa_loss_val, global_step)
            
            # --- Evaluation ---
            if global_step % args.eval_every == 0:
                val_loss, val_acc = evaluate(model, test_loader, criterion)
                writer.add_scalar('Valid/Acc_Base', val_acc, global_step)
                
                log_str = f"Ep {epoch} Step {global_step}: Loss {loss.item():.4f} | Base Acc {val_acc:.4f}"
                
                curr_perf = val_acc
                
                # Evaluate RNSA Refined Model
                if rnsa_active and global_step > 1:
                    _, r_acc = evaluate(refined_model, test_loader, criterion)
                    writer.add_scalar('Valid/Acc_RNSA', r_acc, global_step)
                    log_str += f" | RNSA Acc {r_acc:.4f}"
                    curr_perf = r_acc
                
                print(log_str)
                
                if curr_perf > best_acc:
                    best_acc = curr_perf
                    save_path = os.path.join(out_dir, "best_model.pth")
                    torch.save(model.state_dict(), save_path)
                    print(f"  -> Best model saved! ({best_acc:.4f})")

    print(f"Done. Best Acc: {best_acc:.4f}")
    writer.close()

if __name__ == "__main__":
    train()