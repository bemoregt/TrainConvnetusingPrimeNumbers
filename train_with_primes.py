import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

cudnn.benchmark = True
plt.ion()   # interactive mode

# 데이터 증강과 정규화 설정
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop((244, 244)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation([90, 90]),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((244, 244)),
        transforms.CenterCrop((244, 244)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def get_prime_numbers(n):
    """n보다 작은 소수들의 리스트를 반환"""
    primes = []
    for i in range(2, n):
        is_prime = True
        for j in range(2, int(i ** 0.5) + 1):
            if i % j == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(i)
    return primes

def create_subset_sampler(dataset_size, prime):
    """소수를 이용한 서브셋 인덱스 생성"""
    indices = []
    for i in range(dataset_size):
        if i % prime == 0:
            indices.append(i)
    return torch.utils.data.SubsetRandomSampler(indices)

# 데이터 로딩
data_dir = r"C:\Users\PC\Downloads\hymenoptera_data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                        data_transforms[x])
                 for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes
train_primes = get_prime_numbers(dataset_sizes['train'])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def imshow(inp, title=None):
    """Tensor를 이미지로 표시"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)
    
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    with TemporaryDirectory() as tempdir:
        best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')
        torch.save(model.state_dict(), best_model_params_path)
        best_acc = 0.0

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            # 매 에포크마다 다른 소수 선택
            if train_primes:
                current_prime = train_primes[epoch % len(train_primes)]
                train_sampler = create_subset_sampler(dataset_sizes['train'], current_prime)
                
                train_loader = torch.utils.data.DataLoader(
                    image_datasets['train'],
                    batch_size=16,
                    sampler=train_sampler,
                    num_workers=12
                )
                
                print(f'Using prime number: {current_prime} for sampling')
            
            dataloaders = {
                'train': train_loader,
                'val': torch.utils.data.DataLoader(
                    image_datasets['val'],
                    batch_size=16,
                    shuffle=True,
                    num_workers=12
                )
            }

            current_train_size = len(train_sampler) if train_primes else dataset_sizes['train']
            current_dataset_sizes = {
                'train': current_train_size,
                'val': dataset_sizes['val']
            }

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / current_dataset_sizes[phase]
                epoch_acc = running_corrects.double() / current_dataset_sizes[phase]

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_params_path)

            print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:4f}')

        model.load_state_dict(torch.load(best_model_params_path))
    return model

# 모델 초기화
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)
model_ft = model_ft.to(device)

# 손실 함수와 옵티마이저 설정
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# 모델 학습
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                      num_epochs=25)

# 모델 저장
torch.save(model_ft, "residueNorm_resnet18_9.pth")