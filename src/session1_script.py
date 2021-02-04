
import numpy as np
import sys
sys.path.append(os.path.join('..')) # Include the home dir in our path to be able to import modules in notebooks that are not child of this dir?
import cv2
from utils.imutils import jimshow

def translate(img, x = 0, y = 0):
    height, width = img.shape[:2]
    
    m = np.float64([[1, 0, x],
                    [0, 1, y]])
    return cv2.warpAffine(img, m, (width, height)) # Translates img


def main():
    

if __name__ == "__main__":
    main()