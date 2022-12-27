import os
import math
import pydicom as dicom
import tensorflow as tf
import numpy as np
import h5py
import cv2
import scipy.ndimage

from PIL import Image
from tensorflow.keras.utils import array_to_img 
from skimage.metrics import structural_similarity as compare_ssim


# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        #distance between slices, finds slice tkickness if not availabe
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

# Taking Housenfield Unit into consideration
def get_pixels_hu(slices):
    # read the dicom images, find HU numbers (padding, intercept, rescale), and make a 4-D array, 
    # HU - Hounsfield Unit (HU): measure of radiodensity

    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    try:
        padding = slices[0].PixelPaddingValue
    except:
        padding = 0
    
    image[image == padding] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
        
    return np.array(image, dtype=np.int16)

#write a nmpy array in a dicom image
def write_dicom(slices,arrays,path):
    # array should be between 0-4095
    for i in range(arrays.shape[0]):
        new_slice = slices
        pixel_array = ((arrays[i,:,:,0]+new_slice.RescaleIntercept)/new_slice.RescaleSlope).astype(np.int16)
        new_slice.PixelData = pixel_array.tostring()
        new_slice.save_as(path+'/'+str(i)+'.dcm')

# This function extracts patches from the larger image
def extract_patches(image, patch_size=40,stride=20):
    images_num,h,w = image.shape
    out = np.empty((0,patch_size,patch_size))
    sz = image.itemsize
    shape = ((h-patch_size)//stride+1, (w-patch_size)//stride+1, patch_size,patch_size)
    strides = sz*np.array([w*stride,stride,w,1])

    for d in range (0,images_num):
        patches=np.lib.stride_tricks.as_strided(image[d,:,:], shape=shape, strides=strides)
        blocks=patches.reshape(-1,patch_size,patch_size)
        out=np.concatenate((out,blocks[:,:,:]))
        #print(d)
    
    return out[:,:,:]

def write_hdf5(data,labels, output_filename):
    """
    This function is used to save image data and its label(s) to hdf5 file.
    output_file.h5,contain data and label
    """

    with h5py.File(output_filename, 'w') as h:
        h.create_dataset('data', data=data, shape=data.shape)        
        h.create_dataset('label', data=labels, shape=labels.shape)
        h.close()
 

def read_hdf5(file):
    with h5py.File(file, 'r') as hf:
        data = np.array(hf.get('data'))
        labels = np.array(hf.get('label'))
   
        return data,labels 

def measure_psnr(test_labels, pred_labels):
    diff = test_labels-pred_labels
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    psnr = 20*math.log10(1.0/rmse)
    return psnr

def measure_ssim(test_labels, pred_labels):
    ssim = 0
    for i in range (test_labels.shape[0]):
        ssim += compare_ssim(test_labels[i,:,:,0],pred_labels[i,:,:,0],
                    data_range=pred_labels[i,:,:,0].max()-pred_labels[i,:,:,0].min())
        
    ssim = ssim/test_labels.shape[0]
    return ssim     

# POST PROCESSING TOOLS
#This function streches the gray scale range between min and max bound for better visualization of the details 
def windowing2(image,center,width):
    min_bound = center - width/2
    max_bound = center + width/2
    output = (image-min_bound)/(max_bound-min_bound)
    output[output<0]=0
    output[output>1]=1
    return output 

def convert_to_rgb(img):
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def ROI(img, roi1, roi2, color, thickness):
    bound_box = cv2.rectangle(img, roi1['start'], roi1['end'], color['red'], thickness)
    bound_box = cv2.rectangle(img, roi2['start'], roi2['end'], color['blue'], thickness)
    return bound_box

def crop_resize(img, roi1, roi2, new_size):
    row1, col1 = [roi1['start'][1]+1, roi1['end'][1]], [roi1['start'][0]+1, roi1['end'][0]]
    row2, col2 = [roi2['start'][1]+1, roi2['end'][1]], [roi2['start'][0]+1, roi2['end'][0]]
    cropped1 = img[row1[0]:row1[1], col1[0]:col1[1]]
    cropped2 = img[row2[0]:row2[1], col2[0]:col2[1]]
    resize1 = cv2.resize(cropped1, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    resize2 = cv2.resize(cropped2, (new_size, new_size), interpolation=cv2.INTER_LINEAR)
    return resize1, resize2

def load_img(path):
    img = np.array(Image.open(path))
    return img

def save_img(img, path):
    Image.fromarray(img).save(path)

def extract_roi_results(img, roi1, roi2, color, save_bounding_path, save_crop_path, file_type):
    im = ROI(convert_to_rgb(img), roi1, roi2, color, 2)
    cropped_im1, cropped_im2 = crop_resize(im, roi1, roi2, 256)
    save_img(im, os.path.join(save_bounding_path, file_type))
    save_img(cropped_im1, os.path.join(save_crop_path, '_first_'+file_type))
    save_img(cropped_im2, os.path.join(save_crop_path, '_second_'+file_type))

def extract_img(data):
    img = windowing2((data)*4095-1024,40,400)
    return array_to_img(img)

## DATA PIPELINE TOOLS
def process_scan(dcm_data):
    slice = dicom.read_file(dcm_data)

    img = slice.pixel_array
    img = img.astype(np.int16)

    intercept = slice.RescaleIntercept
    slope = slice.RescaleSlope

    if slope != 1:
        img = slope*img.astype(np.float64)
        img = img.astype(np.int16)

    img += np.int16(intercept)
    img = np.array(img, dtype=np.int16) + 1024
    img = np.expand_dims(img, axis=0)
    img = (img[:,:,:,None]/4095).astype(np.float32)

    return img

def process_predict(pred):
    img = windowing2(pred[0,:,:,:]*4095-1024, 40, 400)
    img = array_to_img(img)
    return img

def display_dcm(dcm_data):
    slice = dicom.read_file(dcm_data)

    img = slice.pixel_array
    img = img.astype(np.int16)

    intercept = slice.RescaleIntercept
    slope = slice.RescaleSlope

    if slope != 1:
        img = slope*img.astype(np.float64)
        img = img.astype(np.int16)

    img += np.int16(intercept)
    img = np.array(img, dtype=np.int16)
    img = np.expand_dims(img, axis=0)
    img = (img[:,:,:,None]).astype(np.float32)
    img = windowing2(img[0,:,:,:], 40, 400)
    img = array_to_img(img)
    return img

if __name__ == '__main__':
    #==============================================================
    # Piglet DATA SET
    #Setting file path for the low and high dose images
    path_low = 'data/Piglet/Paired/Low'
    path_high= 'data/Piglet/Paired/High'

    train_data_path = 'data/processed/train_32_32_piglet_dicom_0.h5'
    test_data_path = 'data/processed/test_32_32_piglet_dicom_0.h5'

    # Loading dicom files from Piglet dataset 
    slices_data_train = load_scan(path_low)
    slices_labels_train = load_scan(path_high)

    #callibrating in terms of HU
    data = get_pixels_hu(slices_data_train)+1024 
    labels = get_pixels_hu(slices_labels_train)+1024

    # Dividing into train (70%) and test(30%) datasets 
    n = len(data)*7//10
    data_train ,data_test = data[:n],data[n:]
    labels_train,labels_test = labels[:n],labels[n:]

    # Extract patches from data (low) and label (high), patch size and stride of 32
    data_patch=extract_patches(data_train,patch_size=32,stride=32)
    labels_patch=extract_patches(labels_train,patch_size=32,stride=32)

    # write in h5 files (piglet)
    write_hdf5(data_patch,labels_patch, train_data_path)
    write_hdf5(data_test,labels_test,test_data_path)

    print('Runtime is finished.')