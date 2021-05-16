import cv2 as cv
import os
import csv
import glob

count = 1
path = 'dataset_test'
with open('dataset_test.csv', 'w', newline='') as file:
    fieldnames = ['filename', 'label']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writerow({'filename': 'filename', 'label' : 'label'})
    for img_name in glob.iglob('dataset_test/*.jpg', recursive=True):
        img_name = img_name.replace('dataset_test\\', '')
        print(img_name)
        #img_name = cv.imread(os.path(path, filename))
        if ('glaucoma' in img_name):   
            writer.writerow({'filename': img_name, 'label': 0})
        elif ('normal' in img_name):  
            writer.writerow({'filename': img_name, 'label': 1})
        elif ('other' in img_name):  
            writer.writerow({'filename': img_name, 'label': 2})
        count = count + 1
