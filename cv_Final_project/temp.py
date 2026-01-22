from PIL import Image
import numpy as np

mask = Image.open('/data/data/user/zq/zqWorkPlace/task/cvTask/groundtruth/1.png').convert('RGB')
mask_np = np.array(mask)
print(np.unique(mask_np.reshape(-1,3), axis=0))
