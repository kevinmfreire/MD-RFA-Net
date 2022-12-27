# import streamlit as st
from network import MD_RFA
from utils import process_scan, process_predict, display_dcm

# sys.path.append('src/')
# from src import network, prep, results, utils

class MLPipeline():
    def __init__(self):
        self.model_weight = 'model_weight/weights_mdrfa_perceptual40_mse30_dsim30.h5'
        self.model = MD_RFA()
        self.model.load_weights(self.model_weight)

    def pre_process(self, data):
        return process_scan(data)

    def predict(self, img):
        [labels_pred,labels_pred, labels_pred]= self.model.predict(img,verbose=2)
        return labels_pred

    def post_process(self, pred):
        return process_predict(pred)

    def display(self, dcm_data):
        return display_dcm(dcm_data)

if __name__ == '__main__':
    data_path = '../data/ml_test/1-278-low.dcm'
    data_path2 = '../../BAF-RESNET/comparative_results/content/thoracic/test_data/test_img0.png'
    model_weight = '../model_weight/weights_mdrfa_perceptual40_mse30_dsim30.h5'

    pipe = MLPipeline()
    img = pipe.pre_process(data_path)
    pred = pipe.predict(img)
    result = pipe.post_process(pred)
    result.save('test_img_pred.png')