from utils import load_scan, get_pixels_hu, extract_patches, write_hdf5

def process_data(path_low, path_high, train_data_path, test_data_path, patch_size):
      for i in range(len(path_high)):
        # Loading dicom files from Piglet dataset 
        slices_data_train = load_scan(path_low[i])
        slices_labels_train = load_scan(path_high[i])

        #callibrating in terms of HU
        data = get_pixels_hu(slices_data_train)+1024 
        labels = get_pixels_hu(slices_labels_train)+1024

        # Dividing into train (70%) and test(30%) datasets 
        n = len(data)*7//10
        data_train ,data_test = data[:n],data[n:]
        labels_train,labels_test = labels[:n],labels[n:]

        # Extract patches from data (low) and label (high), patch size and stride of 32
        data_patch=extract_patches(data_train,patch_size=patch_size,stride=patch_size)
        labels_patch=extract_patches(labels_train,patch_size=patch_size,stride=patch_size)

        # write in h5 files (piglet)
        write_hdf5(data_patch,labels_patch, train_data_path[i])
        write_hdf5(data_test,labels_test,test_data_path[i])


if __name__ == '__main__':

    # Setting paths for all data sets and for saving files
    low_dose_files=['data/Thoracic/Copies/Low Dose', 'datasets/Piglet/Paired/Low', 
                    'data/Head - TCIA - Subject N012/Low Dose Images','datasets/Chest - TCIA - Subject C002/Low Dose Images',
                    'data/Abdomen - TCIA - Subject L058/Low Dose Images']

    full_dose_files=['data/Thoracic/Copies/High Dose', 'datasets/Piglet/Paired/High', 
                    'data/Head - TCIA - Subject N012/Full Dose Images','datasets/Chest - TCIA - Subject C002/Full Dose Images',
                    'data/Abdomen - TCIA - Subject L058/Full Dose Images']

    patch_size = '32_32'
    dataset_type = ['thoracic', 'piglet', 'head_N012', 'chest_C002', 'abdomen_L058']
    train_data_path = ['data/processed/train_{}_{}_dicom_0.h5'.format(patch_size, type_) for type_ in dataset_type]
    test_data_path = ['data/processed/test_{}_{}_dicom_0.h5'.format(patch_size, type_) for type_ in dataset_type]

    folder = ['Thoracic', 'Piglet', 'Head - TCIA - Subject N012', 'Chest - TCIA - Subject C002', 'Abdomen - TCIA - Subject L058']
    train_data_files = ['datasets/{}/train_{}_{}_dicom_0.h5'.format(folder[i], patch_size, dataset_type[i]) for i in range(len(folder))]
    test_data_files = ['datasets/{}/test_{}_{}_dicom_0.h5'.format(folder[i], patch_size, dataset_type[i]) for i in range(len(folder))]

    process_data(low_dose_files, full_dose_files, train_data_path, test_data_path, 32)