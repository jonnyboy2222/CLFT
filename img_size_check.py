import os
import cv2
import numpy as np

data_list = open("/home/john/dev_ws/CLFT/waymo_dataset/visual_run_demo.txt", 'r')
data_cam = np.array(data_list.read().splitlines())
dataroot = './waymo_dataset/'
data_list.close()

i = 1
for path in data_cam:
    cam_path = os.path.join(dataroot, path)

    img = cv2.imread(cam_path)

    cv2.imshow(f"img[{i}]", img)
    print(img.shape)

    i += 1    
        
cv2.waitKey(0)
cv2.destroyAllWindows()            