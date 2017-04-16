import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from scipy.misc import imread
from sklearn.externals import joblib
import time
import random as rand
import numpy as np 
import cv2
import glob
from features import *
import matplotlib.gridspec as gridspec


# Visualise HOG transforms   
def hog_visualise(img, hog_img, title = '', save=True ):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(img)
    plt.title(title)
    plt.subplot(122)
    plt.imshow(hog_img, cmap='Purples')
    plt.title('HOG Visualization')
    plt.show()
    if save:
        fig.savefig("output_images/" + title + "_HOG.png")

#load training set
def load_training_set():
    vehicle_imgs, nonvehicle_imgs = [], []
    vehicle_paths = glob.glob('vehicles/*/*.png')
    nonvehicle_paths = glob.glob('non-vehicles/*/*.png')

    for path in vehicle_paths: vehicle_imgs.append(imread(path))
    for path in nonvehicle_paths: nonvehicle_imgs.append(imread(path))

    vehicle_imgs, nonvehicle_imgs = np.asarray(vehicle_imgs), np.asarray(nonvehicle_imgs)
    
    return vehicle_imgs, nonvehicle_imgs

def save_model(clf,scaler,hog_params,title=""):
    model = {'clf': clf, 'scaler': scaler, 'hog_params': hog_params}
    joblib.dump(model, 'model_' + title + '.pkl')
    

def train(vehicles_features, nonvehicles_features):
    unscaled_x = np.vstack((vehicles_features, nonvehicles_features)).astype(np.float64)

    scaler = StandardScaler().fit(unscaled_x)
    x = scaler.transform(unscaled_x)
    total_vehicles, total_nonvehicles = len(vehicles_features), len(nonvehicles_features)
    y = np.hstack((np.ones(total_vehicles),np.zeros(total_nonvehicles)))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                        random_state = rand.randint(1, 100))
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    return clf, scaler, accuracy


import matplotlib.gridspec as gridspec
        
def show_images(imgs, rows = 6, cols = 3, titles = [], output_file=None, W = 30, H = 30):      
    fig = plt.figure(figsize=(W, H))
    gs = gridspec.GridSpec(rows, cols, wspace=0.0)
    ax = [plt.subplot(gs[i]) for i in range(rows*cols)]
    gs.update(hspace=0)
    for i,im in enumerate(imgs):
        ax[i].imshow(im)
        ax[i].axis('off')
        if(len(titles) > i):
            ax[i].set_title(titles[i])
    if output_file is not None:
        fig.savefig(output_file)
    plt.show()

def area(bbox):
    return (bbox[1][1] - bbox[0][1]) * (bbox[1][0] -bbox[0][0])

def draw_labeled_bboxes(img, labels):
    color_map = [(255, 0, 255),(0, 255, 255),(0, 0, 128),(0, 128, 128),(128, 120, 0),(0, 0, 255)]
    for car_number in range(1, labels[1] + 1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        if area(bbox)> 6400:
            cv2.rectangle(img, bbox[0], bbox[1], color_map[car_number % len(color_map) -1], 6)
    return img


def draw_boxes(img,boxes, color=(0, 0, 255), thick=6):
    img_draw = np.copy(img)
    for (x_ll, y_ll), (x_ur,y_ur) in boxes:
        cv2.rectangle(img_draw, (x_ll, y_ll), (x_ur,y_ur), color, thick)
    return img_draw


