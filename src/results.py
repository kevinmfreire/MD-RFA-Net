import os
from utils import load_img, extract_roi_results

if __name__ == '__main__':

    roi_dict = {'piglet': [{'start':(80, 190), 'end':(144,254)}, {'start':(176, 296), 'end':(240, 360)}],
                    'thoracic': [{'start':(224, 256), 'end':(288, 320)}, {'start':(59, 222), 'end':(123, 286)}],
                    'head_N012': [{'start':(96, 116), 'end':(160, 180)}, {'start':(156, 316), 'end':(220, 380)}],
                    'chest_C002': [{'start':(216, 328), 'end':(280, 392)}, {'start':(12, 170), 'end':(76, 234)}],
                    'abdomen_L058': [{'start':(224, 358), 'end':(282, 422)}, {'start':(208, 166), 'end':(272, 230)}]
                    }

    color_dict = {'red':(255,0,0),
                'blue':(0,0,255)
                }

    path = '../data/results/'                                   
    dataset_dir = ['thoracic', 'piglet', 'head_N012', 'chest_C002', 'abdomen_L058']
    dir_set = ['test_data', 'test_labels', 'test_results']

    save_bounding_path = '../data/results/roi/bounding/'
    save_crop_path = '../data/results/roi/cropped/'

    for dataset in dataset_dir:
        for dir_ in dir_set:
            if dir_ == 'test_data':
                file = 'test_img0.png'
                img_file = os.path.join(path, dataset, dir_, file)
                file_type = '{}_{}'.format(dataset, file)
                im = load_img(img_file)
                extract_roi_results(im, roi_dict[dataset][0], roi_dict[dataset][1], color_dict, save_bounding_path, save_crop_path, file_type)
            elif dir_ == 'test_labels':
                file = 'gt_img0.png'
                img_file = os.path.join(path, dataset, dir_, file)
                file_type = '{}_{}'.format(dataset, file)
                im = load_img(img_file)
                extract_roi_results(im, roi_dict[dataset][0], roi_dict[dataset][1], color_dict, save_bounding_path, save_crop_path, file_type)
            if dir_ == 'test_results':
                file = 'pred_img0.png'
                img_file = os.path.join(path, dataset, dir_, file)
                file_type = '{}_{}'.format(dataset, file)
                im = load_img(img_file)
                extract_roi_results(im[:,:,0], roi_dict[dataset][0], roi_dict[dataset][1], color_dict, save_bounding_path, save_crop_path, file_type)
