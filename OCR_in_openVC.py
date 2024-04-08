import cv2
import numpy as np
import imutils
import keras
import numpy as np
from Token_word import OCR_TOKEN
import time



class OCR_Model_find_litary():
	"""
	Модель компьютерного зрения

	Отладка и шлифовка продолжаются
	"""
	def __init__(self, path=None):
		self.path = path
		self.ITEMS = self.OCR_GET(path)

	def get_items(self):
		return self.ITEMS
	def strips_liters(self, imgs):
		# print(imgs.shape, '|UNIT|')
		staper = []
		h, w = imgs.shape
		cater = w // h
		# print(cater, 'RCATERRR')

		if w < h*2:
			hh = w//2
			lefter = imgs[:, :hh]
			imgs = imgs[:, hh:]
		else:
			lefter = imgs[:, :h]
			imgs = imgs[:, h:]

		staper.append(lefter)
		if imgs.shape[1] > imgs.shape[0] + imgs.shape[0] // 2:
			# imgs = cv2.erode(imgs, None, iterations=2)
			for el in range(cater - 1):
				h = w // round(w / h)
				lef = imgs[:, :h]
				imgs = imgs[:, h:]
				staper.append(lef)
			# 	print(lef.shape)
			# 	cv2.imshow(f'lefter part_|clip|: {el}', lef)
			# 	print(lef.shape)
			#
			# cv2.waitKey()
			# print(cater, '|__CATER__|')

		staper.append(imgs)
		#
		# cv2.imshow('lefter partCL', lefter)
		# cv2.imshow('righterCL', imgs)
		# cv2.waitKey()
		return staper

	def sort_conturs_litera(self, cnt, method='rig-lig'):
		reverse = False
		i = 0
		bound_rect = [cv2.boundingRect(el) for el in cnt]
		(cnt, bound_rect) = zip(*sorted(zip(cnt, bound_rect), key=lambda b:b[1][i], reverse=reverse))
		return cnt
	def Litary_detect(self, image):
		"""Разделение и нахождение букв"""
		img =image.copy()
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (23, 23))
		blacked = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
		trash = cv2.threshold(blacked, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

		conturs,_ = cv2.findContours(trash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cv2.drawContours(image, conturs, -1, (0, 255, 0), 1)

		conturs = self.sort_conturs_litera(conturs)


		word = []
		for cont in conturs:
			x, y, w, h = cv2.boundingRect(cont)

			rel = trash[y: y+h, x: x+w]
			rel = cv2.bitwise_not(rel)
			rel = cv2.dilate(rel, (1, 1), iterations=2)

			# TF_CV(rel)

			if rel.shape[0] > 35:
				"""there"""
				# cv2.imshow('ISNT_REL', rel)
				# cv2.waitKey()

				"""Literaly = rel"""
				rel = imutils.resize(rel, height=80)
				cat = rel.shape[0] / 2
				if rel.shape[1] > rel.shape[0]+cat:
					"""проверка на слитые слова"""
					split_liter = self.strips_liters(rel)
					for litera in split_liter:
						word.append(litera)
				else:
					word.append(rel)
		# 			cv2.imshow('REL', rel)
		# 			print(rel.shape, '|RELSHAPE|')
		# 			cv2.waitKey()
		# 	cv2.destroyAllWindows()
		#
		# cv2.imshow('keyDET', trash)
		# cv2.waitKey()
		# cv2.imshow('keyDET', image)
		# cv2.waitKey()
		# cv2.destroyAllWindows()
		return word






	def Words_detected(self, image):
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
		words = []
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

				# """То с чем дальше работать roi"""
				roi = cv2.resize(roi, (roi.shape[1]*3, roi.shape[0]*3))
				#
				# cv2.imshow('imageROI', roi)
				# cv2.waitKey()

				text_word = self.Litary_detect(roi)
				words.append(text_word)
		#
		# cv2.imshow('Image', image)
		# cv2.imwrite('img.png', image)
		# cv2.waitKey()
		return words


	def buFfer(self, el):
		bsize = 30
		el = cv2.resize(el, (80, 80))
		bord = cv2.copyMakeBorder(
			el,
			top=bsize,
			bottom=bsize,
			left=bsize,
			right=bsize,
			borderType=cv2.BORDER_CONSTANT,
			value=[255, 255, 255]
		)
		for ell in range(2):
			bord = cv2.erode(bord, None, iterations=1)
		bord = cv2.GaussianBlur(bord, (3, 3), 0)
		# cv2.imshow('TESTed', bord)
		el = cv2.resize(bord, (28, 28))
		el = np.expand_dims(el, axis=0)
		return el, bord

	def Pip_main(self, image):
		words = self.Words_detected(image)
		return words

	def OCR_GET(self, file):
		if file == None:
			return None
		model = keras.models.load_model(
			r'C:\Users\Антонио\PycharmProjects\Test_concurs\OCR_models\test_model_ocr_recovVV3.h5')
		w = self.Pip_main(file)
		items = ''
		for l in w:
			for el in l:
				bel, bord = self.buFfer(el)
				detect = model(bel)
				s = np.argmax(detect)
				item = OCR_TOKEN(s).get_lit()
				items += item
			# print(s+1, ' | ',item)
			# cv2.waitKey()
			items += '\n'
			# print(f)
			time.sleep(2)
		return items

def main():
	OCR_Model_find_litary()

if __name__ == '__main__':
	main()