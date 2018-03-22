import os
import random
import numpy as np
import skimage.data
from skimage import io
import skimage.transform as T
import skimage.util as util
import selectivesearch
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input

#for Intersection over Union
from collections import namedtuple
import cv2

import warnings

import xml.etree.ElementTree as ET

#SELECTIVE SEARCH PARAMS
SCALE = 500
SIGMA = 0.2
MIN_SIZE = 225

UoI_THRESHOLD = 0.5

random.seed(2506)

IMG_HEIGHT = 224
IMG_WIDTH = 224

IMAGES_DIR = 'JPEGimages'
LABELS_DIR = 'ImageSets\\Main'
ANNOTATIONS_DIR = 'Annotations'


EXT = '.JPG'
XML = '.xml'

LABELS = ['aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']
#LABELS += 'background'

NBCLASSES = len(LABELS)
NB_IMAGES = 2


class DataGen(object): #PascalVOCDataGeneratorForObjectDetection
    
    def __init__(self, subset, data_path):
        
        assert subset in ['train', 'val', 'trainval', 'test']
        self.subset = subset

        self.data_path = data_path
        self.images_path = os.path.join(self.data_path, 'JPEGImages')
        self.labels_path = os.path.join(self.data_path, 'ImageSets', 'Main')
        
        
        print(self._get_total_number_images())
        
        
        # The id_to_label dict has the following structure
        # key : image's id (e.g. 00084)
        # value : image's label (e.g. [0, 0, 1, 0, ..., 1, 0])
        self.id_to_label = {}

        self.labels = LABELS
        self.nb_classes = len(self.labels) # 20 classes for PascalVOC

        # Get all the images' ids for the given subset
        self.images_ids_in_subset = self._get_images_ids_from_subset(self.subset)
        self.nb_images = len(self.images_ids_in_subset)
        
        #Get a dict of each image and its number of regions
        self.image_dict,self.total_nb_images = self._get_region_image_table(6403);
        
        #public acces dataset and according dict {image:nb_regions}
        
        
        
        self.DataSet = np.asarray(self.images_ids_in_subset).astype(int);
        
        dic  = {k:v for (k,v) in self.image_dict.items() if k in self.DataSet}
        
        self.DataDict = dic
        
        self.TotalDataSet = sum(dic.values())
        
        # Create the id_to_label dict with all the images' ids 
        # but the values are arrays with nb_classes (20) zeros 
        
       # self._initialize_id_to_label_dict()
        
        # Fill the values in the id_to_label dict by putting 1 when
        # the label is in the image given by the key

        #self._fill_id_to_label_dict_with_classes()   


    
    
    def generateRegionsData(self,Display = False,from_image=0,to_image=1): #Be very careful here
        
        with open('annotations/labels_%06d_%06d.txt'%(from_image+1,to_image+1), 'w'): pass
        for i in range(self.from_image+1,self.to_image+1):
            imagePath = self._get_image_file_path(i)
    
            bb_labels,bb_boxes = self._get_ground_truth_bb(i)
            reg_prop,raw_regions = self._get_regions_propsal(imagePath)        

            regions = []
            labels = []
            #Compare every proposed region with defined Pascal VOC regions
            for r,rp in enumerate(reg_prop):
                saved = False
                for l,bb in enumerate(bb_boxes):
                    pred = self._rect_to_bbox(rp[1])
                    gt = bb;
                    UoI = self._bb_intersection_over_union(gt,pred)
                    if(UoI>UoI_THRESHOLD):               
                        if (Display): self._display_UiO(imagePath,gt,pred)
                        if not saved:
                            regions += [rp]
                            labels += [LABELS.index(bb_labels[l])]
                            saved = True
                        else:
                            labels[-1]= LABELS.index(bb_labels[l])
                    else:
                        if not saved:
                            regions += [rp]
                            labels += [20]
                            saved = True
            print("Treating image N°: %06d ..."%i)
            print("\t There is %d regions proposed and %d regions saved"%(len(raw_regions),len(regions)))
            
            for j,rp in enumerate(regions):
                #continue;
                #image = T.resize(rp[0],(IMG_HEIGHT,IMG_WIDTH,3),mode='constant')
                image = rp[0]
                path = "images/%06d_%03d.jpg"%(i,j+1)
                #path = "images/%03d_%s.jpg"%(j,rp[1])
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        io.imsave(path,image)
                        #regions += [{"num":i,"filepath":path,"rect":rp[1]}]
                        with open('annotations/labels_%06d_%06d.txt'%(self.from_image+1,self.to_image+1), 'a') as f:
                            f.write("%06d %d %d\n"%(i,j+1,labels[j]))
                    except:
                        pass

              
        print("Generation Completed");
        

    def _get_ground_truth_bb(self,fileNum):
        filePath = self._get_annotation_file_path(fileNum)
        tree = ET.parse(filePath)
        root = tree.getroot()
        
        labels = []
        bboxes = []

        for child in root:
            if child.tag == 'object':
                labels += [child[0].text]
                for subchild in child:
                    if subchild.tag == 'bndbox':
                        bbox = subchild
                        bboxes += [(int(bbox[0].text),int(bbox[1].text),int(bbox[2].text),int(bbox[3].text))]
        return labels,bboxes

    def _bbox_to_rect(self,rect):
        y = rect[0]
        x = rect[1]
        
        w = rect[2] - y
        h = rect[3] - x
        
        return (x,y,w,h)

    def _rect_to_bbox(self,rect):
        xmin = rect[0]
        ymin = rect[1]
        xmax = rect[0] + rect[2]
        ymax = rect[1] + rect[3]
        
        return (xmin,ymin,xmax,ymax)
    
    
    def _display_UiO(self,imagePath,gt,pred):
        
        image = cv2.imread(imagePath)
 
        # draw the ground-truth bounding box along with the predicted
        # bounding box
        
        cv2.rectangle(image, tuple([gt[0],gt[1]]), 
            tuple([gt[2],gt[3]]), (0, 255, 0), 2)
        cv2.rectangle(image, tuple([pred[0],pred[1]]), 
            tuple([pred[2],pred[3]]), (0, 0, 255), 2)

        # compute the intersection over union and display it
        iou = self._bb_intersection_over_union(gt,pred)
        cv2.putText(image, "IoU: {:.14f}".format(iou), (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        #print("{}: {:.4f}".format(detection.image_path, iou))
        # show the output image
        cv2.imshow("Image", image)
        cv2.waitKey(0)
    
  
    
    def _get_regions_propsal(self,imgpath):
        #reading image from file
        img = io.imread(imgpath)
        #img = T.resize(img,(IMG_HEIGHT,IMG_WIDTH,3),mode='reflect')


        #get region proposal using selective search
        img_lbl, regions = selectivesearch.selective_search(img, scale=SCALE, sigma=SIGMA, min_size=MIN_SIZE)


        #crop the original image according to the regions proposed and store it to a new array
        regions_proposal=list()
        for r in regions:
            rect = r['rect']
            y1 = rect[0]
            y2 = rect[0] + rect[2]
            x1 = rect[1]
            x2 = rect[1] + rect[3]
            cropped_im = img[x1:x2,y1:y2]
            #this is problematic
            #cropped_im = T.resize(cropped_im,(IMG_HEIGHT,IMG_WIDTH,3),mode='constant')
            regions_proposal += [(cropped_im,rect)]
    
        return regions_proposal,regions
    

    def _bb_intersection_over_union(self,boxA, boxB):

        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        #interArea = (xB - xA + 1) * (yB - yA + 1)
        interArea = max(0,(xB - xA + 1)) * max(0,(yB - yA + 1)) #correction

        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        
       

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)

        # return the intersection over union value
        return iou
    
    

    def _get_image_file_path(self,num):
    
        imagesPath = os.path.join(self.data_path,IMAGES_DIR)
            
        imageNumber = num;
        imageName = "%06d" % (imageNumber,)
        imageFile = '%s%s'%(imageName,EXT)

        imageFilePath = os.path.join(imagesPath,imageFile)
    
        return imageFilePath;

    def _get_annotation_file_path(self,num):
        dirPath = os.path.join(self.data_path,ANNOTATIONS_DIR)
            
        fileNumber = num;
        fileName = "%06d" % (fileNumber,)
        annotationFile = '%s%s'%(fileName,XML)

        annotationFilePath = os.path.join(dirPath,annotationFile)
        
        return annotationFilePath
    
    def _get_total_number_images(self):
        with open('annotations/labels.txt','r') as f:
            l = f.read().splitlines()
            return len(l)
        
    def _get_regions_of_image(self, imageId):
        filePath = 'annotations/labels.txt'
        labels = np.loadtxt(filePath).astype(int)
        imageLines = np.where(labels[:,0]==int(imageId))[0]
        regions=[]
        for line in labels[imageLines]:
            regions += [ ["%06d_%03d.jpg"%(line[0],line[1]),line[2]]]
        return regions        
        
    def _id_to_label_dict(self):
        for image_id in self.images_ids_in_subset:
            self.id_to_label[image_id] = np.zeros(self.nb_classes)
        pass
        
    def _label_to_one_hot(self,label):
        label_one_hot = np.zeros(self.nb_classes+1)
        label_one_hot[label]=1;
        return label_one_hot.astype(int);
        
    def _initialize_id_to_label_dict(self):
        for image_id in self.images_ids_in_subset:
            self.id_to_label[image_id] = np.zeros(self.nb_classes)

    def _fill_id_to_label_dict_with_classes(self):
        pass
        """_fill_id_to_label_dict_with_classes
        For each class, the <class>_<subset>.txt file contain the presence information
        of this class in the image
        """
        for i in range(self.nb_classes):
            label = self.labels[i]
            # Open the <class>_<subset>.txt file
            with open(os.path.join(self.labels_path, "%s_%s.txt" % (label, self.subset)), 'r') as f:
                lines = f.read().splitlines()
                for line in lines:
                    splited_line = line.split()
                    image_id = splited_line[0]
                    is_present = int(splited_line[1])
                    if is_present == 1:
                        self.id_to_label[image_id][i] = 1

    def _get_region_image_table(self,last):
        fileLines = np.loadtxt('annotations/labels.txt')
        nbr={}
        for i in range(1,last):
            regions =  np.where(fileLines[:,0]==i)[0]
            nbr[i]=len(regions)
        return nbr,len(fileLines)
    
    
    
    def _get_images_ids_from_subset(self, subset):
        """_get_images_ids_from_subset
        The images' ids are found in the <subset>.txt file in ImageSets/Main
        """
        with open(os.path.join(self.labels_path, subset + '.txt'), 'r') as f:
            images_ids = f.read().splitlines()
            #random.shuffle(images_ids)
        return images_ids[:NB_IMAGES]
    
    
    def get_xy_from_image(self,iid):
        image_id = "%06d"%iid
        img = image.load_img(os.path.join(self.images_path, image_id + EXT), grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))         
        #treat regions instead of whole images
        regions = self._get_regions_of_image(image_id)
        X_batch = []
        Y_batch = []
        for region in regions:
                imgPath = os.path.join('images', region[0])
                img = image.load_img(imgPath, grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))       
                # Cast the Image object to a numpy array and put the channel has the last dimension
                img_arr = image.img_to_array(img, data_format='channels_last')
                X_batch.append(img_arr)       
                y = self._label_to_one_hot(region[1])
                Y_batch.append(y)                    
        # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3) 
        X_batch = np.reshape(X_batch, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
        # resize Y_batch in (None, nb_classes) 
        Y_batch = np.reshape(Y_batch, (-1, self.nb_classes+1))
        # The preprocess consists of substracting the ImageNet RGB means values
        # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L72
        X_batch = preprocess_input(X_batch, data_format='channels_last') 
        return X_batch,Y_batch     
    
    
    def _generate_batch(self, batch_size=32):
        nb_batches = int(len(self.images_ids_in_subset) / batch_size) + 1
        nb_batches = 1
        size = len(self.images_ids_in_subset)
        # Before each epoch we shuffle the images' ids
        random.shuffle(self.images_ids_in_subset)
        for i in range(nb_batches):
                # We first get all the images' ids for the next batch
                current_bach = self.images_ids_in_subset[i*batch_size:(i+1)*size]
                #X_batch = []
                #Y_batch = []
                for image_id in current_bach:
                    print(image_id)
                    # Load the image and resize it. We get a PIL Image object 
                    img = image.load_img(os.path.join(self.images_path, image_id + EXT), grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))      
                    
                    #treat regions instead of whole images
                    regions = self._get_regions_of_image(image_id)
                    nb_mini_batches = int(len(regions) / batch_size) + 1
                    for j in range(nb_mini_batches):
                        current_mini_batch = regions[j*batch_size:(j+1)*batch_size]
                        X_batch = []
                        Y_batch = []
                        for region in current_mini_batch:
                            imgPath = os.path.join('images', region[0])
                            img = image.load_img(imgPath, grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))       
                            # Cast the Image object to a numpy array and put the channel has the last dimension
                            img_arr = image.img_to_array(img, data_format='channels_last')
                            X_batch.append(img_arr)       
                            y = self._label_to_one_hot(region[1])
                            Y_batch.append(y)
                    
                        # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3) 
                        X_batch = np.reshape(X_batch, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
                        # resize Y_batch in (None, nb_classes) 
                        Y_batch = np.reshape(Y_batch, (-1, self.nb_classes+1))
                        # The preprocess consists of substracting the ImageNet RGB means values
                        # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L72
                        X_batch = preprocess_input(X_batch, data_format='channels_last') 
                        print(X_batch.shape)
                        print(Y_batch.shape)
    
    def flow2(self, batch_size=32):
        """flow
        This is a generator which load the images and preprocess them on the fly
        When calling next python build in function, it returns a batch with a given size
        with a X_batch of size (None, IMG_HEIGHT, IMG_WIDTH, 3)
        and a Y_batch of size (None, nb_classes)
        The first dimension is the batch_size if there is enough images left otherwise 
        it will be less

        :param batch_size: the batch's size
        """
        nb_batches = int(len(self.images_ids_in_subset) / batch_size) + 1
        
        nb_batches = 1
        size = len(self.images_ids_in_subset)
        
        while True:
            # Before each epoch we shuffle the images' ids
            random.shuffle(self.images_ids_in_subset)
            for i in range(nb_batches):
                # We first get all the images' ids for the next batch
                current_bach = self.images_ids_in_subset[i*batch_size:(i+1)*size]
                #X_batch = []
                #Y_batch = []
                for image_id in current_bach:
                    # Load the image and resize it. We get a PIL Image object 
                    img = image.load_img(os.path.join(self.images_path, image_id+EXT),grayscale=False,target_size=(IMG_HEIGHT, IMG_WIDTH))      
                    #treat regions instead of whole images
                    regions = self._get_regions_of_image(image_id)
                    nb_mini_batches = int(len(regions) / batch_size) + 1
                    for j in range(nb_mini_batches):
                        current_mini_batch = regions[j*batch_size:(j+1)*batch_size]
                        X_batch = []
                        Y_batch = []
                        for region in current_mini_batch:
                            imgPath = os.path.join('images', region[0])
                            img = image.load_img(imgPath, grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))       
                            # Cast the Image object to a numpy array and put the channel has the last dimension
                            img_arr = image.img_to_array(img, data_format='channels_last')
                            X_batch.append(img_arr)       
                            y = self._label_to_one_hot(region[1])
                            Y_batch.append(y)
                    
                        # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3) 
                        X_batch = np.reshape(X_batch, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
                        # resize Y_batch in (None, nb_classes) 
                        Y_batch = np.reshape(Y_batch, (-1, self.nb_classes+1))
                        # The preprocess consists of substracting the ImageNet RGB means values
                        # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L72
                        X_batch = preprocess_input(X_batch, data_format='channels_last')
                        if(len(X_batch)>0):
                            yield(X_batch, Y_batch)    

                
    def flow(self, batch_size=32):
        """flow
        This is a generator which load the images and preprocess them on the fly
        When calling next python build in function, it returns a batch with a given size
        with a X_batch of size (None, IMG_HEIGHT, IMG_WIDTH, 3)
        and a Y_batch of size (None, nb_classes)
        The first dimension is the batch_size if there is enough images left otherwise 
        it will be less

        :param batch_size: the batch's size
        """
        nb_batches = int(len(self.images_ids_in_subset) / batch_size) + 1
        while True:
            # Before each epoch we shuffle the images' ids
            random.shuffle(self.images_ids_in_subset)
            for i in range(nb_batches):
                # We first get all the images' ids for the next batch
                current_bach = self.images_ids_in_subset[i*batch_size:(i+1)*batch_size]
                X_batch = []
                Y_batch = []
                for image_id in current_bach:
                    # Load the image and resize it. We get a PIL Image object 
                    img = image.load_img(os.path.join(self.images_path, image_id + EXT), grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))                                    
                    #treat regions instead of whole images
                    regions = self._get_regions_of_image(image_id)
                    
                    for region in regions:
                        imgPath = os.path.join('images', region[0])
                        img = image.load_img(imgPath, grayscale=False, target_size=(IMG_HEIGHT, IMG_WIDTH))       
                        # Cast the Image object to a numpy array and put the channel has the last dimension
                        img_arr = image.img_to_array(img, data_format='channels_last')
                        X_batch.append(img_arr)       
                        y = self._label_to_one_hot(region[1])
                        Y_batch.append(y)
                    
                # resize X_batch in (batch_size, IMG_HEIGHT, IMG_WIDTH, 3) 
                X_batch = np.reshape(X_batch, (-1, IMG_HEIGHT, IMG_WIDTH, 3))
                # resize Y_batch in (None, nb_classes) 
                Y_batch = np.reshape(Y_batch, (-1, self.nb_classes+1))
                # The preprocess consists of substracting the ImageNet RGB means values
                # https://github.com/keras-team/keras/blob/master/keras/applications/imagenet_utils.py#L72
                X_batch = preprocess_input(X_batch, data_format='channels_last')
                yield(X_batch, Y_batch)                   
                
                
                
                
                
                
