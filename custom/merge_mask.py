import cv2
import numpy as np 
import sys



def merge(a,b,c,d):
    pos = cv2.imread(a)
    neg = cv2.imread(b)
    msk = cv2.imread(c)[:,:, 0]

    canvas = np.zeros(pos.shape)
    canvas[msk == 255, :] = pos[msk == 255,:]
    canvas[msk == 0, :] = neg[msk == 0,:]

    cv2.imwrite(str(d), canvas)



if __name__ == "__main__":
    merge(*sys.argv[1:])