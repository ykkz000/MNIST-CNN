import numpy as np

import models
import sys
import matplotlib.pyplot as plt


def draw(config: models.ModelConfig):
    train_losses = np.load('losses/{}_train.loss.npy'.format(config.model_name))
    val_losses = np.load('losses/{}_val.loss.npy'.format(config.model_name))
    train_accs = np.load('losses/{}_train.acc.npy'.format(config.model_name))
    val_accs = np.load('losses/{}_val.acc.npy'.format(config.model_name))
    train_times = np.load('losses/{}_train.time.npy'.format(config.model_name))
    epochs = np.arange(len(train_losses))
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    plt.subplots_adjust(hspace=0.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('{} Loss'.format(config.model_name))
    ax1.grid(True, linestyle='--', color='gray', alpha=0.3)
    ax1.plot(epochs, train_losses, color='red', label='train_loss')
    ax1.plot(epochs, val_losses, color='blue', label='val_loss')
    ax1.legend()
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('{} Accuracy'.format(config.model_name))
    ax2.grid(True, linestyle='--', color='gray', alpha=0.3)
    ax2.plot(epochs, train_accs, color='red', label='train_acc')
    ax2.plot(epochs, val_accs, color='blue', label='val_acc')
    ax2.legend()
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Time')
    ax3.set_title('{} Train Time'.format(config.model_name))
    ax3.grid(True, linestyle='--', color='gray', alpha=0.3)
    ax3.plot(epochs, train_times, color='red', label='train_time')
    ax3.legend()
    plt.show()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python draw.py model_name')
    config = models.get_config_from_json('config/config_{}.json'.format(sys.argv[1]))
    draw(config)
