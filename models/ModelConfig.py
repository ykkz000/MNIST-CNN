import json


class ModelConfig:
    def __init__(self, model_name, epochs, batch_size, learning_rate, weight_decay, step, gamma, seed):
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.step = step
        self.gamma = gamma
        self.seed = seed

    def __str__(self):
        return 'ModelConfig(model_name={}, epochs={}, batch_size={}, learning_rate={}, weight_decay={}, step={}, gamma={}, seed={})'.format(self.model_name, self.epochs, self.batch_size, self.learning_rate, self.weight_decay, self.step, self.gamma, self.seed)


def get_config_from_json(json_path: str):
    with open(json_path, 'r') as f:
        config = json.load(f)
    return ModelConfig(config['model_name'], config['epochs'], config['batch_size'], config['learning_rate'], config['weight_decay'], config['step'], config['gamma'], config['seed'])
