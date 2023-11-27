#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore', category=UserWarning)


# In[2]:


from duckduckgo_search import ddg_images # pip install duckduckgo_search
from fastcore.all import * # conda install -c fastai fastai

from fastdownload import download_url
from fastai.vision.all import *


# In[3]:


# A function that search images.
def search_images(term, max_images=30):
    print(f"Searching for '{term}'")
    return L(ddg_images(term, max_results=max_images)).itemgot('image') # L is a listlike type specific to fastcore


# In[4]:


bird_dest = 'bird.jpg'
bird_urls = search_images('bird photos', max_images=1) # Relies on ddg, if error just try again.
download_url(bird_urls[0], bird_dest, show_progress=True)

im = Image.open(bird_dest)
im.to_thumb(256,256)


# In[5]:


# Search, download and show a picture of a forest.
forest_dest = 'forest.jpg'
forest_urls = search_images('forest photos', max_images=1)
download_url(forest_urls[0], forest_dest, show_progress=True)

im = Image.open(forest_dest)
im.to_thumb(256,256)


# In[6]:


searches = 'forest','bird'
path = Path('bird_or_not')
from time import sleep

for o in searches:
    dest = (path/o)
    dest.mkdir(exist_ok=True, parents=True)
    download_images(dest, urls=search_images(f'{o} photo'))
    sleep(10)  # Pause between searches to avoid over-loading server
    download_images(dest, urls=search_images(f'{o} sun photo'))
    sleep(10)
    download_images(dest, urls=search_images(f'{o} shade photo'))
    sleep(10)
    resize_images(path/o, max_size=400, dest=path/o)


# In[7]:


# Removed images that did not download correctly.
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
len(failed)


# In[8]:


# Prepare the training data. Both the training set and the validation set.
# DataBlock is fastai's way to do this task.
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock), # Input is images, Output is categories (bird / forest).
    get_items=get_image_files, # get_image_files return a list of all the images in the given path.
    splitter=RandomSplitter(valid_pct=0.2, seed=42), # Use 20% of the data as validation set.
    get_y=parent_label, # The name of the parent folder (bird or forest) will be used as category labels.
    item_tfms=[Resize(192, method='squish')] # Resize image by squishing (not cropping) before training.
).dataloaders(path, bs=32)

dls.show_batch(max_n=6)


# In[9]:


# Train and tune our model.
learn = vision_learner(dls, resnet18, metrics=error_rate) # Resnet18 is a widely used, fast, cv model.
learn.fine_tune(3) # FastAI use best practices for fine tuning a pre-trained model.


# In[10]:


# Use our model by passing it the first picture that we downloaded.
is_bird,_,probs = learn.predict(PILImage.create('bird.jpg'))
print(f"This is a: {is_bird}.")
print(f"Probability it's a bird: {probs[0]:.4f}")


# In[ ]: