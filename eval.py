import models
import sys
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


def test(model, config: models.ModelConfig, device, test_data):
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=config.batch_size, shuffle=True)
    model.eval()
    sum_acc = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)
            acc = torch.sum(predicted == labels)
            sum_acc += acc.item()
    print('Accuracy for Test :{:.2f}%'.format(sum_acc * 100 / len(test_data)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python eval.py model_name')
    config = models.get_config_from_json('config/config_{}.json'.format(sys.argv[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('checkpoints/{}_best.pth'.format(config.model_name), map_location=device)
    trans = transforms.Compose([
        transforms.ToTensor()
    ])
    test_data = datasets.FashionMNIST('dataset', train=False, transform=trans, download=True)
    test(model, config, device, test_data)
