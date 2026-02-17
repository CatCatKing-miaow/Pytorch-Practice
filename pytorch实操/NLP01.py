import torch
import torch.nn as nn
import torch.optim as optim

sentences = ["i love this movie", "this movie is good", "fantastic result", 
             "i hate this movie", "this movie is bad", "terrible result"]
labels = [1, 1, 1, 0, 0, 0]  # 1=Pos, 0=Neg

# 构建词表 (Vocabulary)
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
word2idx = {w: i for i, w in enumerate(word_list)}
vocab_size = len(word2idx)

# 把句子转换成 Tensor
def make_data(sentence):
    idxs = [word2idx[w] for w in sentence.split()]
    return torch.tensor(idxs, dtype=torch.long)

inputs = [make_data(s) for s in sentences]
targets = torch.tensor(labels, dtype=torch.float)

class TextClassifier(nn.Module):
    def __init__(self):
        super(TextClassifier, self).__init__()
        # Embedding层
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=10)
        # LSTM层
        self.lstm = nn.LSTM(input_size=10, hidden_size=6)
        # Linear层
        self.linear = nn.Linear(in_features=6, out_features=1)
        self.sigmoid = nn.Sigmoid() # 把结果压到 0-1 之间

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds.view(len(x), 1, -1))
        last_hidden= lstm_out[-1] 
        # 全连接层分类
        out = self.linear(last_hidden)
        return torch.sigmoid(out)

model = TextClassifier()

criterion = nn.BCELoss()  
optimizer = optim.Adam(model.parameters(), lr=0.01)

print("开始训练...")
for epoch in range(500):  # 训练500轮
    total_loss = 0
    for i in range(len(inputs)):
        #  清空梯度
        model.zero_grad()
        #  前向传播
        output = model(inputs[i])
        #  计算 Loss
        loss = criterion(output.view(-1), targets[i].view(-1))
        #  反向传播 & 更新参数
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

print("\n=== 测试结果 ===")
test_sent = "i love result" 
test_vec = make_data(test_sent)
predict = model(test_vec)
print(f"输入句子: '{test_sent}'")
print(f"预测得分: {predict.item():.4f} ")

test_sent2 = "terrible movie"
test_vec2 = make_data(test_sent2)
predict2 = model(test_vec2)
print(f"输入句子: '{test_sent2}'")
print(f"预测得分: {predict2.item():.4f} ")