#!/usr/bin/env python
# coding: utf-8

# In[3]:

#to seperate images into test and trainset 
#to seperate annotations into test and train 
import os
import shutil
import random

# Paths
images_folder =  r"image path"
annotations_folder = r"annotation path"
output_folder = r"output folder path"

 #Output directories
train_images_dir = os.path.join(output_folder, "train/images")
train_annotations_dir = os.path.join(output_folder, "train/annotations")
test_images_dir = os.path.join(output_folder, "test/images")
test_annotations_dir = os.path.join(output_folder, "test/annotations")

#Create output directories
os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_annotations_dir, exist_ok=True)
os.makedirs(test_images_dir, exist_ok=True)
os.makedirs(test_annotations_dir, exist_ok=True)

# List all images and annotations
image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith(".xml")]
print(f"Found {len(image_files)} images and {len(annotation_files)} annotations.")
# Pair images with annotations
for i in range(len(image_files)):
    image_files[i]=image_files[i].replace("jpg.jpg","jpg")
paired_files = []


for image_file in image_files:
    annotation_file = image_file + ".xml"
    
    if annotation_file in annotation_files:
        paired_files.append([image_file, annotation_file])
print(f"Number of paired files: {len(paired_files)}")
print(paired_files[0])
# Shuffle and split
random.shuffle(paired_files)
split_ratio = 0.8
split_index = int(len(paired_files) * split_ratio)

train_pairs = paired_files[:split_index]
test_pairs = paired_files[split_index:]
print(train_pairs[1][1])
 #Copy files to train and test directories
for image_file, annotation_file in train_pairs:
    
    shutil.copy(os.path.join(images_folder, image_file.replace(".jpg",".jpg.jpg")), train_images_dir)
    shutil.copy(os.path.join(annotations_folder, annotation_file), train_annotations_dir)

for image_file, annotation_file in test_pairs:
    shutil.copy(os.path.join(images_folder, image_file.replace(".jpg",".jpg.jpg")), test_images_dir)
    shutil.copy(os.path.join(annotations_folder, annotation_file), test_annotations_dir)

print("Data split completed.")


# In[4]:


# step-2: read xml files
# from each xml file we need to extract
# filename, size(width, height), object(name, xmin, xmax, ymin, ymax)
from xml.etree import ElementTree as et # parse information from XML

def extract_text(filename):
    tree = et.parse(filename)
    root = tree.getroot()

    # extract filename
    image_name = root.find('filename').text
    # width and height of the image
    width = root.find('size').find('width').text
    height = root.find('size').find('height').text
    objs = root.findall('object')
    parser = []
    for obj in objs:
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        xmax = bndbox.find('xmax').text
        ymin = bndbox.find('ymin').text
        ymax = bndbox.find('ymax').text
        parser.append([image_name, width, height, name,xmin,xmax,ymin,ymax])
        
    return parser


# In[5]:


annotations_folder = r"C:\Users\anush\Downloads\archive (4)\annotations\annotations"

annotation_files = [f for f in os.listdir(annotations_folder) if f.endswith(".xml")]


# In[7]:


from functools import reduce

annot1=[]
for i in range(len(annotation_files)):
    
    annot1.append(os.path.join(annotations_folder,annotation_files[i]))
print(annot1[0])
parser_all = list(map(extract_text,annot1))
data = reduce(lambda x, y : x+y,parser_all)


# In[8]:


import pandas as pd
df = pd.DataFrame(data,columns = ['filename','width','height','name','xmin','xmax','ymin','ymax'])


# In[9]:


df.head()


# In[10]:


df['name'].value_counts()


# In[11]:


annot_train=[]
for i in range(len(train_pairs)):
    
    annot_train.append(os.path.join(annotations_folder,train_pairs[i][1]))
#annotation_files=(os.path.join(annotations_folder,train_pairs[][1])


# In[12]:


parser_all = list(map(extract_text,annot_train))


# In[13]:


from functools import reduce
data1 = reduce(lambda x, y : x+y,parser_all)


# In[14]:


df_train = pd.DataFrame(data1,columns = ['filename','width','height','name','xmin','xmax','ymin','ymax'])


# In[15]:


df_train.head()


# In[16]:


df_train.info()


# In[17]:


cols = ['width','height','xmin','xmax','ymin','ymax']
df_train[cols] = df_train[cols].astype(float)
df_train.info()


# In[18]:


# center x, center y
df_train['center_x'] = ((df_train['xmax']+df_train['xmin'])/2)/df_train['width']
df_train['center_y'] = ((df_train['ymax']+df_train['ymin'])/2)/df_train['height']
# w 
df_train['w'] = (df_train['xmax']-df_train['xmin'])/df_train['width']
# h 
df_train['h'] = (df_train['ymax']-df_train['ymin'])/df_train['height']


# In[19]:


# label encoding

def label_encoding(x):
    labels = {'keyboard':0, 'mouse':1, 'laptop':2, 'monitor':3, 'mobile':4}
    return labels[x]
df_train['id'] = df_train['name'].apply(label_encoding)


# In[20]:


df_train.head()


# In[34]:


cols = ['filename','id','center_x','center_y', 'w', 'h']
groupby_obj_train = df_train[cols].groupby('filename')


# In[24]:


folder_path=r"C:\Users\anush\Downloads\archive (4)\output_folder\train\annot_text"
for filename, group in groupby_obj_train:
    text_filename = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
    group.set_index('filename').to_csv(text_filename, sep=' ', index=False, header=False)


# preparing test annotation data in text format
# 

# In[25]:


annot_test=[]
for i in range(len(test_pairs)):
    
    annot_test.append(os.path.join(annotations_folder,test_pairs[i][1]))


# In[26]:


parser_all = list(map(extract_text,annot_test))


# In[27]:


from functools import reduce
data1 = reduce(lambda x, y : x+y,parser_all)#converting 3d to 2d(see parser_all o/p 3 brackets ididhu 2 agutte)


# In[28]:


df_test = pd.DataFrame(data1,columns = ['filename','width','height','name','xmin','xmax','ymin','ymax'])


# In[29]:


cols = ['width','height','xmin','xmax','ymin','ymax']
df_test[cols] = df_test[cols].astype(float)
df_test.info()


# In[30]:


# center x, center y
df_test['center_x'] = ((df_test['xmax']+df_test['xmin'])/2)/df_test['width']
df_test['center_y'] = ((df_test['ymax']+df_test['ymin'])/2)/df_test['height']
# w 
df_test['w'] = (df_train['xmax']-df_test['xmin'])/df_test['width']
# h 
df_test['h'] = (df_test['ymax']-df_test['ymin'])/df_test['height']


# In[31]:


def label_encoding(x):
    labels = {'keyboard':0, 'mouse':1, 'laptop':2, 'monitor':3, 'mobile':4}
    return labels[x]
df_test['id'] = df_test['name'].apply(label_encoding)


# In[33]:


cols = ['filename','id','center_x','center_y', 'w', 'h']
groupby_obj_test = df_test[cols].groupby('filename')


# In[35]:


folder_path=r"C:\Users\anush\Downloads\archive (4)\output_folder\test\annot_text"
for filename, group in groupby_obj_test:
    text_filename = os.path.join(folder_path, os.path.splitext(filename)[0] + '.txt')
    group.set_index('filename').to_csv(text_filename, sep=' ', index=False, header=False)


# In[ ]:




