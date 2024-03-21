# this is a demo code for computing the average intensity of proteins of all images (save as .csv)
import numpy as np
import pandas as pd
import imageio
from IPython.display import display, Image
import time
from tqdm import tqdm
import os

# --------- specify these args ----------
img_dir = 'train/images'  # dir that saves your images
n_train = 225  # number of images
n_protein = 52  # number of proteins
# ---------------------------------------

avg_list = []
for i in tqdm(np.arange(1, n_train+1), total=n_train, desc="Processing"):
    img_file_name = f'{i}.tiff'
    path = os.path.join(img_dir, img_file_name)
    img = imageio.v2.imread(path)
    avg = np.mean(img, axis=(1,2))
    avg_list.append([i] + avg.tolist())
avg_df = pd.DataFrame(avg_list, columns = ['id'] + ['protein' + str(i) for i in range(1, 52+1)])
avg_df.to_csv('avg_intensity.csv', index=False)

avg_df.head()
