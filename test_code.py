import cv2
import numpy as np
import imutils

k = ["\\0.jpeg","\\1.jpeg","\\2.jpeg","\\3.jpeg","\\4.png",
	 "\\5.jpeg","\\6.jpeg",'\\7.jpeg', '\\8.jpeg', '\\9.jpeg']
file = r"C:\Datasets\Pasports"



def strips_liters(imgs):
	print(imgs.shape, '|UNIT|')
	staper = []
	cater = imgs.shape[1] // imgs.shape[0]
	h, w = imgs.shape
	lefter = imgs[:, :h]
	imgs = imgs[:, h:]

	staper.append(lefter)
	if imgs.shape[1] > imgs.shape[0] + imgs.shape[0] // 2:
		for el in range(cater - 1):
			lef = imgs[:, :h]
			imgs = imgs[:, h:]
			staper.append(lef)
			cv2.imshow(f'lefter part_|clip|: {el}', lef)

		cv2.waitKey()
		print(cater, '|__CATER__|')

	staper.append(imgs)

	cv2.imshow('lefter part', lefter)
	cv2.imshow('righter', imgs)
	cv2.waitKey()
	return staper

def Litary_detect(image):
	"""Разделение букв"""
	img =image.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
	blacked = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
	trash = cv2.threshold(blacked, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

	conturs,_ = cv2.findContours(trash, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(image, conturs, -1, (0, 255, 0), 1)

	word = []
	for cont in conturs:
		x, y, w, h = cv2.boundingRect(cont)
		rel = trash[y: y+h, x: x+w]
		if rel.shape[0] > 35:
			print(rel.shape)

			"""Literaly = rel"""
			rel = imutils.resize(rel, height=80)
			cat = rel.shape[0] / 2
			if rel.shape[1] > rel.shape[0]+cat:
				"""проверка на слитые слова"""
				split_liter = strips_liters(rel)
				for litera in split_liter:
					word.append(litera)
			else:
				word.append(rel)
				cv2.imshow('REL', rel)
				print(rel.shape, '|RELSHAPE|')
				cv2.waitKey()
		cv2.destroyAllWindows()

	cv2.imshow('key', trash)
	cv2.waitKey()
	cv2.imshow('key', image)
	cv2.waitKey()
	cv2.destroyAllWindows()




def Words_detected(image):
	image =cv2.imread(image)
	image = imutils.resize(image, height=1000)
	img = image.copy()
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (3, 3), 0)

	recetKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
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
			pX = int((x+w) * 0.01)
			pY = int((y+h) * 0.02)
			(x, y) = (x - pX, y - pY)
			(w, h) = (w +(pX * 2), h + (pY * 2))
			roi = image[y : y + h, x:x + w].copy()
			rois.append(roi)
			# cv2.rectangle(image, (x, y), (x + w, y+h), (0, 255, 0), 2)

			# """То с чем дальше работать roi"""
			roi = cv2.resize(roi, (roi.shape[1]*3, roi.shape[0]*3))

			cv2.imshow('image', roi)
			cv2.waitKey()
			Litary_detect(roi)

	cv2.imshow('Image', image)
	cv2.imwrite('img.png', image)
	cv2.waitKey()


def Pip_main(image):
	Words_detected(image)

def main():
	for i in k:
		print(i)
		# if i == '\\4.png':
		Pip_main(file + i)


if __name__ == '__main__':
	main()