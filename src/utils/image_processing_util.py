import numpy as np
from PIL import Image
import os
def load_image(image_path):
    image=Image.open(image_path)
    image=image.resize((30,30))
    image=np.array(image)
    return image
def arrange_data(folder):
    data=[]
    labels=[]
    num_of_class=43

    for i in range(num_of_class):
        path=os.path.join(folder,str(i))
        images=os.listdir(path)
        for img in images:
            image_path=os.path.join(path,img)
            try:
                image=load_image(image_path)
                data.append(image)
                labels.append(i)
            except:
                print("Error while loading Image....")
    data=np.array(data)
    labels=np.array(labels)
    return data,labels

