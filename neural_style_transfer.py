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

blob = cv2.dnn.blobFromImage(image, 1.0, (w,h), 
		(103.939, 116.779, 123.680), swapRB = False, crop = False)
net.setInput(blob)
start = time.time()
output = net.forward()
end = time.time()

#reshaping output tensor
output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680

output /= 255.0
output = output.transpose(1,2,0)

# show information on how long inference took
print("[INFO] neural style transfer took {:.4f} seconds".format(
	end - start))
 
cv2.imshow("Input", image)
cv2.imshow("Output", output)
cv2.waitKey(0)