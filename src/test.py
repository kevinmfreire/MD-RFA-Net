import os
import math
import pandas as pd
import numpy as np

from PIL import Image
from tensorflow.keras.preprocessing import image
from keras.optimizers import Adam
from network import MD_RFA
from utils import read_hdf5, measure_psnr, measure_ssim, extract_img
from loss import psnr_loss, dssim_loss, ssim_loss, ResNet50V2Model
from keras.losses import MeanSquaredError

def convert_to_png(test, label, pred, results_path):
    dataset_dir = ['thoracic', 'piglet', 'head_N012', 'chest_C002', 'abdomen_L058']
    dir_set = ['test_data', 'test_labels', 'test_results']
    for i in range(len(pred_results)):
        num_images, _, _, _ = pred_results[i].shape
        for dir in dir_set:
            for j in range(num_images):
                if dir == 'test_data':
                    data = test[i][j,:,:,:]
                    img = extract_img(data)
                    path = '../{}/{}/{}/test_img{}.png'.format(results_path, dataset_dir[i], dir, j)
                    img.save(path)
                elif dir == 'test_labels':
                    data = label[i][j,:,:,:]
                    img = extract_img(data)
                    path = '../{}/{}/{}/gt_img{}.png'.format(results_path, dataset_dir[i], dir, j)
                    img.save(path)
                if dir == 'test_results':
                    data = pred[i][j,:,:,:]
                    img = extract_img(data)
                    path = '../{}/{}/{}/pred_img{}.png'.format(results_path, dataset_dir[i], dir, j)
                    img.save(path)

def get_test_data(dataset_type):
    testing_data, testing_labels = [], []
    for key, value in dataset_type.items():
        test_address = '../data/processed/test_32_32_{}_dicom_0.h5'.format(key, value)
        data_test,labels_test = read_hdf5(test_address)
        data_test = (data_test[:,:,:,None]/4095).astype(np.float32)
        labels_test = (labels_test[:,:,:,None]/4095).astype(np.float32)
        testing_data.append(data_test)
        testing_labels.append(labels_test)

    return testing_data, testing_labels

def test_model(model, data, labels, model_weight_address, dataset_type, model_arch, weight, feature_extract):
    results_dict = {}
    pred_results = []
    for i, type_ in enumerate(dataset_type):

        perceptual = ResNet50V2Model()
        mse = MeanSquaredError()

        loss = [perceptual.perceptual_loss, mse, dssim_loss]
        loss_weights = [40,30,30]

        # Test
        ADAM=Adam(learning_rate=0.0002, beta_1=0.01, beta_2=0.999)
        model.compile(optimizer=ADAM,loss=loss, loss_weights=loss_weights,metrics=[psnr_loss, ssim_loss])

        model.load_weights(model_weight_address)

        [labels_pred,labels_pred, labels_pred]= model.predict(data[i],batch_size=8,verbose=1)

        psnr = measure_psnr(labels[i], labels_pred)
        ssim = measure_ssim(labels[i], labels_pred)

        results_dict[type_] = {'psnr':psnr, 'ssim':ssim}
        pred_results.append(labels_pred)

        labels_test_3 = np.concatenate((labels[i],labels[i],labels[i]),axis=-1)

        loss=model.evaluate(x=data[i],y=[labels_test_3,labels_test_3,labels_test_3], batch_size=8,verbose=1)

    df = pd.DataFrame(results_dict)
    df.to_csv('model_{}_{}_results_{}_{}_{}.csv'.format(model_arch, feature_extract, weight[0], weight[1], weight[2]))
    print(df)

    return results_dict, pred_results


if __name__ == '__main__':

    dataset_type = {'Thoracic':'thoracic', 'Piglet':'piglet', 'Head':'head_N012', 'Chest':'chest_C002', 'Abdomen':'abdomen_L058'}
    weight_address = 'model/weights/weights_mdrfa_perceptual40_mse30_dsim30.h5'

    model=MD_RFA()
    testing_data, testing_labels = get_test_data(dataset_type)

    results, pred_results = test_model(model, testing_data, testing_labels, weight_address, dataset_type)
    results_path = '../data/results'
    convert_to_png(testing_data, testing_labels, pred_results, results_path)


#w_labels=windowing2((labels_test)*4095+1024,40,400)
#w_data=windowing2((data_test)*4095+1024,40,400)
#w_pred=windowing2((labels_pred)*4095+1024,40,400)

##show one test results
# plt.imshow(data_test[40,:,:,0], cmap='gray')
# plt.show()
# plt.figure()
# plt.imshow((labels_pred)[40,:,:,0], cmap='gray')
# plt.show()
# plt.figure()
# plt.imshow(labels_test[40,:,:,0], cmap='gray')
# plt.show()

# print('The PSNR is:', psnr_edge_p)
# print('The SSIM is:', ssim_edge_p)