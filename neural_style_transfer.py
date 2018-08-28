import argparse
import numpy
import cv2
import time
import imutils

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required = True,
				help = "path to neural style transfer model")
ap.add_argument("-i", "--image", required = True,
				help = "input image for the model")
args = vars(ap.parse_args())

print("[INFO] Importing Neural Style Model ")
net = cv2.dnn.readNetFromTorch(args["model"])

#reading the image and resizing it to 600px
image = cv2.imread(args["image"])
image = imutils.resize(image, width = 600)
(h, w) = image.shape[:2]

