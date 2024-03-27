import cv2
import numpy as np
from PIL import Image, ImageEnhance
import imutils

k = ["\\0.jpeg","\\1.jpeg","\\2.jpeg","\\3.jpeg","\\4.png",
	 "\\5.jpeg","\\6.jpeg",'\\7.jpeg', '\\8.jpeg', '\\9.jpeg']
file = r"C:\Datasets\Pasports"
def image_contrast(im):
	im = Image.open(im)
	enh = ImageEnhance.Contrast(im)
	factor = 1
	im_outer = enh.enhance(factor)
	itoger = np.array(im_outer)

	# im_outer.show()

	return itoger

def Rectangle(image):
	img = image.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	recetKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
	blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, recetKernel)

	gradinX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
	gradinX = np.absolute(gradinX)
	(minVal, maxVal) = (np.min(gradinX), np.max(gradinX))
	gradinX = (255 * ((gradinX - minVal) / (maxVal - minVal))).astype('uint8')

	gradinX = cv2.morphologyEx(gradinX, cv2.MORPH_CLOSE, recetKernel)
	treshold = cv2.threshold(gradinX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	treshold = cv2.morphologyEx(treshold, cv2.MORPH_DILATE, np.ones((1, 2), np.uint8))

	p = int(image.shape[1] * 0.05)
	treshold[:, 0:p]= 0
	treshold[:, image.shape[1] - p:] = 0
	cnt, ierarh = cv2.findContours(treshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(img, cnt, -1, (0, 255, 0),1, cv2.LINE_AA)
	cnt = sorted(cnt, key = cv2.contourArea, reverse=True)
	rois = []
	for c in cnt:
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w/float(h)
		crWitgh = w/float(gray.shape[1])
		if ar > 2 and crWitgh > 0.05:
			pX = int((x+w) * 0.03)
			pY = int((y+h) * 0.03)
			(x, y) = (x - pX, y - pY)
			(w, h) = (w +(pX * 2), h + (pY * 2))
			roi = image[y : y + h, x:x + w].copy()
			rois.append(roi)
			cv2.rectangle(image, (x, y), (x + w, y+h), (0, 255, 0), 2)




	cv2.imshow('Image', image)
	cv2.imwrite('img.png', image)
	cv2.waitKey()


def Pip_main(image):
	Rectangle(image_contrast(image))

def main():
	for i in k:
		print(i)
		Pip_main(file + i)
		# if i == '\\4.png':
		# 	Pip_main(file + i)

if __name__ == '__main__':
	main()