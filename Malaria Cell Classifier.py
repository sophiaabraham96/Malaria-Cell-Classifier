
# coding: utf-8

# # Creating a Classifier for Malaria Cells 
# *by Sophia Abraham* 
# 
# In an attempt to understand different applications of computer vision, here is my attempt at creatifing a classifier to help people by detecting whether the cells contain Malaria! How exciting! 
# 
# [Image Data Set](https://ceb.nlm.nih.gov/repositories/malaria-datasets/)

# ## Creating the folders with the images

# Creating directories for the folders and designating appropriate names for the labeled images.

# In[10]:


folder = 'Uninfected'


# In[7]:


folder = 'Parasitized'


# And then, create a directory for the files 
# 

# In[11]:


path = Path('data/cell_images')
dest = path/folder 
dest.mkdir(parents=True, exist_ok=True)


# In[5]:


from fastai import *
from fastai.vision import * 


# In[12]:


path.ls()


# And then the images were uploaded from computer to the folder in Jupyter Notebooks 
# 
# I just uploaded a large zip file and and then unzipped it below:

# In[19]:


import zipfile
zip_ref = zipfile.ZipFile("data/cell_images.zip", 'r')
zip_ref.extractall("data")
zip_ref.close()


# ## View Data 

# In[20]:


np.random.seed(42)
data = ImageDataBunch.from_folder('data/cell_images', train=".", valid_pct=0.2,
                                 ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# In[21]:


data.classes


# In[22]:


data.show_batch(rows = 3, figsize=(7,8))


# In[23]:


data.classes, data.c, len(data.train_ds), len(data.valid_ds)


# ## Train the Model 
# 
# First. create the convolutional neural net 

# In[24]:


learn = create_cnn(data, models.resnet34, metrics=error_rate)


# In[25]:


learn.fit_one_cycle(4)


# In[26]:


learn.save('stage-1')


# In[27]:


learn.unfreeze()


# In[28]:


learn.lr_find()


# In[29]:


learn.recorder.plot()


# In[30]:


learn.fit_one_cycle(2, max_lr=slice(3e-7,3e-4))


# In[31]:


learn.save('stage-2')


# ## Interpretation

# In[32]:


learn.load('stage-2')


# In[33]:


interp = ClassificationInterpretation.from_learner(learn)


# In[34]:


interp.plot_confusion_matrix()


# ## Cleaning Up 
# 

# In[35]:


from fastai.widgets import * 


# In[37]:


ds , idxs = DatasetFormatter().from_toplosses(learn, ds_type=DatasetType.Valid)


# In[39]:


ImageCleaner(ds, idxs, path)

