import matplotlib.pyplot as plt
from PIL import Image
import os
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg


def vietOCR_prediction(input):
    current_directory = os.getcwd()
    absolute_path = os.path.join(current_directory, 'Model\data', 'my_model.pth')
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = 'cpu'
    config['weights'] = absolute_path
    config['cnn']['pretrained'] = True
    detector = Predictor(config)
    return detector.predict(input)
