import models.Models
import models.resnet

def get_model(model_name, channels, labels):
    if model_name == 'FinalModel':
        return models.FinalModel(channels, labels)
    elif model_name == 'SmallerResNet':
        return models.SmallerResNet(channels, labels)
    elif model_name == 'EluResNet':
        return models.EluResNet(channels, labels)
    elif model_name == 'GeluResNet':
        return models.GeluResNet(channels, labels)
    elif model_name == 'AdaptiveResNet':
        return models.AdaptiveResNet(channels, labels)
    elif model_name == 'RationalResNet':
        return  models.RationalResNet(channels, labels)
    elif model_name == 'ResNet18':
        return models.resnet.ResNet18(channels, labels)
    elif model_name == 'ResNet34':
        return models.resnet.ResNet34(channels, labels)
    elif model_name == 'ResNet50':
        return models.resnet.ResNet50(channels, labels)
    elif model_name == 'ResNet101':
        return models.resnet.ResNet101(channels, labels)
    elif model_name == 'ResNet152':
        return models.resnet.ResNet152(channels, labels)
    return None
