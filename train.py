import models
import random
import time
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


def setup_seed(config: models.ModelConfig):
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.backends.cudnn.deterministic = True


def train(model, config: models.ModelConfig, device, train_data, val_data):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=config.gamma)
    train_loader = DataLoader(dataset=train_data, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=config.batch_size, shuffle=True)
    best_acc = 0
    best_epoch = 0
    print("Start training")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_times = []
    epochs = []
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0.6)
    plt.ion()
    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        start_time = time.time()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            predicted = torch.argmax(outputs, dim=1)
            acc = torch.sum(predicted == labels)
            train_loss += loss.item()
            train_acc += acc.item()
        scheduler.step()
        end_time = time.time()
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                predicted = torch.argmax(outputs, dim=1)
                acc = torch.sum(predicted == labels)
                val_loss += loss.item()
                val_acc += acc.item()
        train_loss_epoch = train_loss / len(train_data)
        train_acc_epoch = train_acc / len(train_data)
        val_loss_epoch = val_loss / len(val_data)
        val_acc_epoch = val_acc / len(val_data)
        train_losses.append(train_loss_epoch)
        val_losses.append(val_loss_epoch)
        train_accs.append(train_acc_epoch)
        val_accs.append(val_acc_epoch)
        train_times.append(end_time - start_time)
        epochs.append(epoch)
        print(
            'epoch:{} | time:{:.2f} | train_loss:{:.8f} | train_acc:{:.4f}% | val_loss:{:.8f} | val_acc:{:.4f}%'.format(
                epoch,
                end_time - start_time,
                train_loss_epoch,
                train_acc_epoch * 100,
                val_loss_epoch,
                val_acc_epoch * 100))
        ax1.clear()
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('{} Loss'.format(config.model_name))
        ax1.plot(epochs, train_losses, color='red', label='train_loss')
        ax1.plot(epochs, val_losses, color='blue', label='val_loss')
        ax1.grid(True, linestyle='--', color='gray', alpha=0.3)
        ax2.clear()
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('{} Accuracy'.format(config.model_name))
        ax2.plot(epochs, train_accs, color='red', label='train_acc')
        ax2.plot(epochs, val_accs, color='blue', label='val_acc')
        ax2.grid(True, linestyle='--', color='gray', alpha=0.3)
        plt.pause(0.1)
        if val_acc_epoch >= best_acc:
            best_acc = val_acc_epoch
            best_epoch = epoch
            torch.save(model, 'checkpoints/{}_best.pth'.format(config.model_name))
        print('Best Accuracy for Validation :{:.4f}% at epoch {:d}'.format(best_acc * 100, best_epoch))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    plt.ioff()
    plt.show()
    torch.save(model, 'checkpoints/{}_last.pth'.format(config.model_name))
    np.save('losses/{}_train.loss'.format(config.model_name), np.array(train_losses))
    np.save('losses/{}_val.loss'.format(config.model_name), np.array(val_losses))
    np.save('losses/{}_train.acc'.format(config.model_name), np.array(train_accs))
    np.save('losses/{}_val.acc'.format(config.model_name), np.array(val_accs))
    np.save('losses/{}_train.time'.format(config.model_name), np.array(train_times))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python train.py model_name')
    config = models.get_config_from_json('config/config_{}.json'.format(sys.argv[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Config:', config)
    setup_seed(config)
    model = models.get_model(config.model_name, 1, 10).to(device)
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    train_val_data = datasets.FashionMNIST('dataset', train=True, transform=trans, download=True)
    train_data, val_data = torch.utils.data.random_split(train_val_data, [55000, 5000])
    train(model, config, device, train_data, val_data)
