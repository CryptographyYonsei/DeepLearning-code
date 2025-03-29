import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import matplotlib.pyplot as plt


# 1. 데이터셋 클래스 정의: 시뮬레이션 사이드 채널 데이터 생성
class SimulatedSideChannelDataset(data.Dataset):
    def __init__(self, num_samples=2000, seq_length=1000):
        """
        시뮬레이션 사이드 채널 데이터를 생성하는 클래스.
        - 전체 샘플 수의 절반은 '정상' 신호, 나머지 절반은 '공격' 신호로 생성.
        - 정상 신호: 일정 주파수의 사인파 + 약간의 잡음.
        - 공격 신호: 정상 신호에 특정 구간(예: 40%~60%)에 이상 오프셋(예: +0.5)을 추가.
        """
        self.num_samples = num_samples
        self.seq_length = seq_length

        # 데이터와 레이블 초기화
        self.X = np.zeros((num_samples, seq_length, 1), dtype=np.float32)
        self.y = np.zeros(num_samples, dtype=np.int64)

        # 시간 벡터 생성 (0~1초 구간)
        t = np.linspace(0, 1, seq_length)

        for i in range(num_samples):
            # 기본 사인파 파라미터 (주파수, 진폭) 약간의 변동 포함
            freq = np.random.uniform(5, 7)
            amplitude = np.random.uniform(0.8, 1.2)
            signal = amplitude * np.sin(2 * np.pi * freq * t)
            # 정규분포 잡음 추가 (평균 0, 표준편차 0.1)
            noise = np.random.normal(0, 0.1, seq_length)
            signal += noise

            # 전체 샘플의 절반은 정상, 절반은 공격으로 설정
            if i < num_samples // 2:
                label = 0  # 정상 신호
            else:
                label = 1  # 공격 신호: 특정 구간에 이상 오프셋 추가
                start = int(seq_length * 0.4)
                end = int(seq_length * 0.6)
                signal[start:end] += 0.5  # 이상 신호 (오프셋 추가)

            self.X[i, :, 0] = signal
            self.y[i] = label

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 2. 모델 정의 (1D CNN)
class SideChannelCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SideChannelCNN, self).__init__()
        # 첫 번째 합성곱층: 입력 채널 1, 출력 채널 32, 커널 크기 3, 패딩 1 (출력 길이 유지)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)  # 풀링: 시퀀스 길이를 1/2로 줄임
        # 두 번째 합성곱층: 입력 32, 출력 64, 커널 크기 3, 패딩 1
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        # 전결합층: 두 번 풀링하면 길이는 seq_length/4, 여기선 1000/4 = 250
        self.fc1 = nn.Linear(64 * 250, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_length, 1) → CNN 입력은 (batch_size, channels, seq_length)
        x = x.transpose(1, 2)
        x = torch.relu(self.conv1(x))  # (batch_size, 32, seq_length)
        x = self.pool(x)  # (batch_size, 32, seq_length/2)
        x = torch.relu(self.conv2(x))  # (batch_size, 64, seq_length/2)
        x = self.pool(x)  # (batch_size, 64, seq_length/4) → (batch_size, 64, 250)
        x = x.view(x.size(0), -1)  # flatten: (batch_size, 64*250)
        x = torch.relu(self.fc1(x))  # (batch_size, 128)
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, num_classes)
        return x


# 3. 하이퍼파라미터 및 데이터로더 설정
batch_size = 32
num_epochs = 10
learning_rate = 0.001

# 시뮬레이션 데이터셋 생성 (정상과 공격 데이터 포함)
dataset = SimulatedSideChannelDataset(num_samples=2000, seq_length=1000)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 4. 모델, 손실 함수, 옵티마이저 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SideChannelCNN(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 5. 학습 루프
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

# 6. 시각화: 정상 신호와 공격 신호 예제 확인
import matplotlib.pyplot as plt

# 정상 신호 예시 (데이터셋 앞부분의 샘플들, label 0)
plt.figure(figsize=(10, 4))
for i in range(3):
    signal, label = dataset[i]
    plt.plot(signal, label=f'Sample {i} (Normal)')
plt.xlabel('Time Step')
plt.ylabel('Signal Value')
plt.title('Normal Side-Channel Signals')
plt.legend()
plt.show()

# 공격 신호 예시 (데이터셋 후반의 샘플들, label 1)
plt.figure(figsize=(10, 4))
for i in range(2000 // 2, 2000 // 2 + 3):
    signal, label = dataset[i]
    plt.plot(signal, label=f'Sample {i} (Attack)')
plt.xlabel('Time Step')
plt.ylabel('Signal Value')
plt.title('Attack Side-Channel Signals')
plt.legend()
plt.show()

