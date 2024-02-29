import numpy as np
import skimage
import skimage.io as io
from skimage.transform import rescale
import scipy.io as scio
# import distortion_model
import distortion_model as distortion_model
import argparse
import os
from skimage import io
import cv2
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument("--types", nargs='+', default=['barrel',  'pincushion'], 
                    help="List of distortion types. Options: ['fisheye', 'barrel', 'rotation', 'shear', 'wave', 'pincushion', 'projective', 'barrel_projective', 'pincushion_projective']")
parser.add_argument("--sourcedir", type=str, default=r'E:\Code_LRL\GeoProj\data_generation\pic', help='original images path')
parser.add_argument("--datasetdir", type=str, default='test', help='generated distorted images path')
parser.add_argument("--trainnum", type=int, default=5, help='number of the training set')
parser.add_argument("--testnum", type=int, default=0, help='number of the test set')
args = parser.parse_args()

if not os.path.exists(args.datasetdir):
    os.mkdir(args.datasetdir)

trainDisPath = args.datasetdir + '/train_distorted'
trainUvPath  = args.datasetdir + '/train_flow'
testDisPath = args.datasetdir + '/test_distorted'
testUvPath  = args.datasetdir + '/test_flow'

paths = [trainDisPath, trainUvPath, testDisPath, testUvPath]
for path in paths:
    os.makedirs(path, exist_ok=True)

def save_data(img, u, v, types, k, trainFlag):
    
    file_name = f"{types}_{str(k).zfill(6)}"
    dir_path = trainDisPath if trainFlag else testDisPath
    uv_dir_path = trainUvPath if trainFlag else testUvPath

    save_img_path = os.path.join(dir_path, f"{file_name}.jpg")
    save_mat_path = os.path.join(uv_dir_path, f"{file_name}.mat")

    io.imsave(save_img_path, img)
    scio.savemat(save_mat_path, {'u': u, 'v': v})

def crop_range(OriImg, height, width):
    disImg = np.array(np.zeros(OriImg.shape), dtype = np.uint8)
    u = np.array(np.zeros((OriImg.shape[0],OriImg.shape[1])), dtype = np.float32)
    v = np.array(np.zeros((OriImg.shape[0],OriImg.shape[1])), dtype = np.float32)
    
    cropImg = np.array(np.zeros((int(height/2),int(width/2),3)), dtype = np.uint8)
    crop_u  = np.array(np.zeros((int(height/2),int(width/2))), dtype = np.float32)
    crop_v  = np.array(np.zeros((int(height/2),int(width/2))), dtype = np.float32)

    xmin = int(width*1/4)
    xmax = int(width*3/4 - 1)
    ymin = int(height*1/4)
    ymax = int(height*3/4 - 1)
    return disImg, u, v, cropImg, crop_u, crop_v, xmin, xmax, ymin, ymax

def generate_Fusion(types1, types2, k, trainFlag):
    
    print(types1, types2, str(k).zfill(6), trainFlag)

    parameters1 = distortion_model.distortionParameter(types1)
    parameters2 = distortion_model.distortionParameter(types2)
    
    OriImg = io.imread('%s%s%s%s' % (args.sourcedir, '/', str(k).zfill(6), '.jpg'))
    
    if OriImg.ndim == 2:  
        OriImg = np.stack((OriImg,)*3, axis=-1)
        
    if types1 == 'pincushion':
        width, height = 256, 256
        temImg = cv2.resize(OriImg, (width, height), interpolation=cv2.INTER_LINEAR)

        ScaImg = skimage.img_as_ubyte(temImg)
        
        padImg = np.array(np.zeros((ScaImg.shape[0] + 1,ScaImg.shape[1] + 1, 3)), dtype = np.uint8)
        padImg[0:height, 0:width, :] = ScaImg[0:height, 0:width, :]
        padImg[height, 0:width, :] = ScaImg[height - 1, 0:width, :]
        padImg[0:height, width, :] = ScaImg[0:height, width - 1, :]
        padImg[height, width, :] = ScaImg[height - 1, width - 1, :]

        disImg = np.array(np.zeros(ScaImg.shape), dtype = np.uint8)
        u = np.array(np.zeros((ScaImg.shape[0],ScaImg.shape[1])), dtype = np.float32)
        v = np.array(np.zeros((ScaImg.shape[0],ScaImg.shape[1])), dtype = np.float32)

        for i in range(width):
            for j in range(height):
                
                xu1, yu1 = distortion_model.distortionModel(types1, i, j, width, height, parameters1)
                xu, yu = distortion_model.distortionModel(types2, xu1, yu1, width, height, parameters2)
                
                if (0 <= xu <= width - 1) and (0 <= yu <= height - 1):

                    u[j][i] = xu - i
                    v[j][i] = yu - j
                    
                    Q11 = padImg[int(yu), int(xu), :]
                    Q12 = padImg[int(yu), int(xu) + 1, :]
                    Q21 = padImg[int(yu) + 1, int(xu), :]
                    Q22 = padImg[int(yu) + 1, int(xu) + 1, :]
                    
                    disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                    Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                    Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                    Q22*(xu - int(xu))*(yu - int(yu))
        
        save_data(disImg, u, v, types, k, trainFlag)
    
    else:   
        width, height  = 512, 512 
        disImg, u, v, cropImg, crop_u, crop_v, xmin, xmax, ymin, ymax = crop_range(OriImg, height, width)

        for i in range(width):
            for j in range(height):
                
                xu1, yu1 = distortion_model.distortionModel(types1, i, j, width, height, parameters1)
                xu, yu = distortion_model.distortionModel(types2, xu1, yu1, width, height, parameters2)
                
                if (0 <= xu < width - 1) and (0 <= yu < height - 1):

                    u[j][i] = xu - i
                    v[j][i] = yu - j
                
                    Q11 = OriImg[int(yu), int(xu), :]
                    Q12 = OriImg[int(yu), int(xu) + 1, :]
                    Q21 = OriImg[int(yu) + 1, int(xu), :]
                    Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]
                    
                    disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                    Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                    Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                    Q22*(xu - int(xu))*(yu - int(yu))

                    if(xmin <= i <= xmax) and (ymin <= j <= ymax):
                        cropImg[j - ymin, i - xmin, :] = disImg[j,i,:]
                        crop_u[j - ymin, i - xmin] = u[j,i]
                        crop_v[j - ymin, i - xmin] = v[j,i]
        
        save_data(cropImg, crop_u, crop_v, f"{types1}_{types2}", k, trainFlag)

def generatedata(types, k, trainFlag):
    
    print(types, str(k).zfill(6), trainFlag)
    
    width  = 512
    height = 512

    parameters = distortion_model.distortionParameter(types)
    
    OriImg = io.imread('%s%s%s%s' % (args.sourcedir, '/', str(k).zfill(6), '.jpg'))
    if OriImg.ndim == 2:  
        OriImg = np.stack((OriImg,)*3, axis=-1)

    disImg, u, v, cropImg, crop_u, crop_v, xmin, xmax, ymin, ymax = crop_range(OriImg, height, width)

    for i in range(width):
        for j in range(height):
            
            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)
            
            if (0 <= xu < width - 1) and (0 <= yu < height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j

                Q11 = OriImg[int(yu), int(xu), :]
                Q12 = OriImg[int(yu), int(xu) + 1, :]
                Q21 = OriImg[int(yu) + 1, int(xu), :]
                Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]
                
                disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                 Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                 Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                 Q22*(xu - int(xu))*(yu - int(yu))

                            
                if(xmin <= i <= xmax) and (ymin <= j <= ymax):
                    cropImg[j - ymin, i - xmin, :] = disImg[j,i,:]
                    crop_u[j - ymin, i - xmin] = u[j,i]
                    crop_v[j - ymin, i - xmin] = v[j,i]
                    
    save_data(cropImg, crop_u, crop_v, types, k, trainFlag)    
        
def generatepindata(types, k, trainFlag):
    
    print(types, str(k).zfill(6), trainFlag)
    
    width  = 256
    height = 256

    parameters = distortion_model.distortionParameter(types)
    
    OriImg = io.imread('%s%s%s%s' % (args.sourcedir, '/', str(k).zfill(6), '.jpg'))
    
    if len(OriImg.shape) == 2:
        OriImg = skimage.color.gray2rgb(OriImg)

    temImg = cv2.resize(OriImg, (width, height), interpolation=cv2.INTER_LINEAR)

    ScaImg = skimage.img_as_ubyte(temImg)
    
    padImg = np.array(np.zeros((ScaImg.shape[0] + 1,ScaImg.shape[1] + 1, 3)), dtype = np.uint8)
    padImg[0:height, 0:width, :] = ScaImg[0:height, 0:width, :]
    padImg[height, 0:width, :] = ScaImg[height - 1, 0:width, :]
    padImg[0:height, width, :] = ScaImg[0:height, width - 1, :]
    padImg[height, width, :] = ScaImg[height - 1, width - 1, :]

    disImg = np.array(np.zeros(ScaImg.shape), dtype = np.uint8)
    u = np.array(np.zeros((ScaImg.shape[0],ScaImg.shape[1])), dtype = np.float32)
    v = np.array(np.zeros((ScaImg.shape[0],ScaImg.shape[1])), dtype = np.float32)

    for i in range(width):
        for j in range(height):
            
            xu, yu = distortion_model.distortionModel(types, i, j, width, height, parameters)
            
            if (0 <= xu <= width - 1) and (0 <= yu <= height - 1):

                u[j][i] = xu - i
                v[j][i] = yu - j
                
                Q11 = padImg[int(yu), int(xu), :]
                Q12 = padImg[int(yu), int(xu) + 1, :]
                Q21 = padImg[int(yu) + 1, int(xu), :]
                Q22 = padImg[int(yu) + 1, int(xu) + 1, :]
                
                disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                 Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                 Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                 Q22*(xu - int(xu))*(yu - int(yu))
    
    save_data(disImg, u, v, types, k, trainFlag)

def fisheye(types, k, trainFlag):
    
    print("fisheye", str(k).zfill(6), trainFlag)
    
    OriImg = io.imread('%s%s%s%s' % (args.sourcedir, '/', str(k).zfill(6), '.jpg'))

    if len(OriImg.shape) == 2:
        OriImg = np.repeat(OriImg[:, :, np.newaxis], 3, axis=2)
            
    height, width = OriImg.shape[:2]
    
    disImg, u, v, cropImg, crop_u, crop_v, xmin, xmax, ymin, ymax = crop_range(OriImg, height, width)
    
    center_x, center_y = width / 2, height / 2
    k_params = distortion_model.random_k()
    
    for i in range(width):
        for j in range(height):
    
            r_d, r_c = distortion_model.fisheye_distortion(i, j, k_params, center_x, center_y)
            
            xu = i + (i - center_x) * (r_c / r_d) if r_d != 0 else i
            yu = j + (j - center_y) * (r_c / r_d) if r_d != 0 else j

            if (0 <= xu < width - 1) and (0 <= yu < height - 1):
                
                u[j][i] = xu - i
                v[j][i] = yu - j
                
                Q11 = OriImg[int(yu), int(xu), :]
                Q12 = OriImg[int(yu), int(xu) + 1, :]
                Q21 = OriImg[int(yu) + 1, int(xu), :]
                Q22 = OriImg[int(yu) + 1, int(xu) + 1, :]
                
                disImg[j,i,:] = Q11*(int(xu) + 1 - xu)*(int(yu) + 1 - yu) + \
                                 Q12*(xu - int(xu))*(int(yu) + 1 - yu) + \
                                 Q21*(int(xu) + 1 - xu)*(yu - int(yu)) + \
                                 Q22*(xu - int(xu))*(yu - int(yu))
                                 
                if(xmin <= i <= xmax) and (ymin <= j <= ymax):
                    cropImg[j - ymin, i - xmin, :] = disImg[j,i,:]
                    crop_u[j - ymin, i - xmin] = u[j,i]
                    crop_v[j - ymin, i - xmin] = v[j,i]
    
    save_data(cropImg, crop_u, crop_v, types, k, trainFlag)
        
               
types_list = args.types

train_range = range(args.trainnum)
test_range = range(args.trainnum, args.trainnum + args.testnum)

for types in types_list:
    
    if types in ['barrel_projective', 'pincushion_projective']: 
        types1, types2 = types.split('_')
        for k in train_range:
            generate_Fusion(types1, types2, k, trainFlag=True)
        for k in test_range:
            generate_Fusion(types1, types2, k, trainFlag=False)

    elif types in ['barrel', 'rotation', 'shear', 'wave']: 
        for k in train_range:
            generatedata(types, k, trainFlag=True)
        for k in test_range:
            generatedata(types, k, trainFlag=False)
            
    elif types in ['pincushion', 'projective']: 
        for k in train_range:
            generatepindata(types, k, trainFlag=True)
        for k in test_range:
            generatepindata(types, k, trainFlag=False)
    
    elif types == 'fisheye':
        for k in train_range:
            fisheye(types, k, trainFlag=True)
        for k in test_range:
            fisheye(types, k, trainFlag=False)
            
    else:
        print("The input distortion type is incorrect!")
