from typing import Any

import keras
import cv2
import numpy as np
from cv2 import Mat, UMat
from numpy import ndarray, dtype, generic

from Token_word import OCR_TOKEN
import imutils
from imutils import contours


def TES(img, item=None):
	if item != None:
		print(f'|ITEM|:{item}')
	cv2.imshow(f'{img.shape}|TEST READ:', img)
	cv2.waitKey()
	cv2.destroyAllWindows()

class Doc_Read():
	def __init__(self, doc=None):
		self.Doc = doc
		self.Model = keras.models.load_model(r'C:\Users\Антонио\PycharmProjects\Test_concurs\OCR_models\test_model_ocr_recovVV3.h5')
		self.image = cv2.imread(self.Doc)

		self.KernelXshare = cv2.getStructuringElement(cv2.MORPH_RECT, [15, 1])
		self.KernelUp = cv2.getStructuringElement(cv2.MORPH_RECT, [2, 10])
		self.KernelUpOne = cv2.getStructuringElement(cv2.MORPH_RECT, [1, 20])

		self.KernelEllipse_litera = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [5, 5])

		self.KernelHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [20, 20])
	def bord_app(self,item, size =10, top = None, bottom=None, left = None, right = None):
		if size == None:
			s1 = top
			s2 = bottom
			s3 = left
			s4 = right
		elif size != None:
			s1 = size
			s2 = size
			s3 = size
			s4 = size
		item = cv2.copyMakeBorder(
			item,
			top=s1,
			bottom=s2,
			left=s3,
			right=s4,
			value=(255, 255, 255),
			borderType=cv2.BORDER_CONSTANT
		)
		return item
	def find(self, image):
		litary = ''
		itemstest = 'top = 20, bottom= 20, left= 40, right=40'

		img = self.bord_app(image, size=None, top = 30, bottom= 50, left= 50, right=40)
		imgg = img.copy()

		img = cv2.erode(img, self.KernelEllipse_litera, iterations=1)
		# cv2.imshow('ims', img)
		img = cv2.threshold(img, 143, 255, cv2.THRESH_BINARY)[1]

		img = cv2.GaussianBlur(img, (3, 3), 0)
		img = cv2.resize(img, (28, 28))
		# cv2.imshow('images', img)
		num = self.Model(np.expand_dims(img, axis=0))
		num = np.argmax(num)
		litary += OCR_TOKEN(num).get_lit()


		# print(litary, '||', num)
		# TES(imgg)
		return litary


	def strip_literaNUM(self, image):
		literas = ''
		img = image.copy()
		w1, w2 = img.shape
		cat = w2//2
		lef = img[:, :cat-2]
		img = img[:, cat-2:]
		l = (lef, img)
		for i in l:
			literas += self.find(i)
		return literas

	def Strip_one(self, image):
		lit = ''
		image = imutils.resize(image, height=250)
		img = image.copy()

		img = cv2.GaussianBlur(img, (3,3), 0)

		img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY)[1]

		# cv2.imshow('img', img)
		img = cv2.erode(img, self.KernelUp, iterations=4)
		img = cv2.erode(img, self.KernelUpOne, iterations=15)

		# cv2.imshow('imgf', img)
		img = self.bord_app(img, size=30)
		image = self.bord_app(image, size=30)

		# TES(img, '|ELLIps|')

		cont,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cont = self.dock_sort_list(cont)

		for con in cont:

			x, y, w, h = cv2.boundingRect(con)
			img = image[y:y+h, x:x+w]

			e1, e2 = img.shape

			# cat = 280
			cat = e1/2 + e1/4

			if 250 < e1 < 289:
				if e2 > cat:
					# print('strip')
					lit += self.strip_literaNUM(img)
				else:
					# print('NOstrip')
					lit += self.find(img)
		return lit
	def dock_sort_list(self, cnt, typer='right_ligth'):
		rev = False
		if typer == 'right_ligth':
			typer = 0
		if typer == 'Up_down':
			typer = 1

		Bound = [cv2.boundingRect(p) for p in cnt]
		(cnt, Bound) = zip(*sorted(zip(cnt, Bound), key = lambda b: b[1][typer], reverse=rev))
		return cnt


	def read_litera(self, word):
		texts = ''
		res1 = imutils.resize(word, height=78)
		res = cv2.GaussianBlur(res1, (3, 3), 0)
		res = cv2.erode(res, self.KernelUpOne, iterations=20)
		res = self.bord_app(res, size=20)
		img_org = self.bord_app(res1.copy(), size= 20)

		_,Trash =cv2.threshold(res, 190, 200, cv2.THRESH_BINARY)
		cont,_ =cv2.findContours(Trash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cont = self.dock_sort_list(cont)

		for ind, con in enumerate(cont[1:]):
			x, y, w, h =cv2.boundingRect(con)
			img = img_org[y:y+h, x:x+w]
			img = imutils.resize(img, height=205)
			w1, w2 = img.shape

			if 70 < w2 < 390:
				litera = self.Strip_one(img)

				texts += litera
		return texts

	def TEST_read_litera(self, word):
		texts = ''
		res1 = imutils.resize(word, height=78)
		cv2.imshow('res1', res1)
		res = cv2.threshold(res1, 130, 255, cv2.THRESH_BINARY)[1]
		cv2.imshow('ress', res)

		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
		res = cv2.dilate(res, kernel=kernel, iterations=1)
		res = cv2.erode(res, self.KernelUpOne, iterations=10)
		TES(res, '|LLL|')

		res = self.bord_app(res, size=20)
		img_org = self.bord_app(res1.copy(), size=20)

		cont, _ = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cont = self.dock_sort_list(cont)
		for ind, con in enumerate(cont[1:]):
			x, y, w, h = cv2.boundingRect(con)
			img = img_org[y:y + h, x:x + w]
			img = imutils.resize(img, height=205)

			w1, w2 = img.shape
			if 70 < w2 < 390:
				litera = self.Strip_one(img)

				texts += litera
		return texts

	# def read_line(self, line):
	# 	liner = ''
	#
	# 	luxx = cv2.erode(line.copy(), self.KernelUp, iterations=4)
	# 	cv2.GaussianBlur(luxx, (3, 3), 0)
	#
	# 	line = self.bord_app(line, size = 20)
	# 	luxx = self.bord_app(luxx, size = 20)
	#
	# 	cv2.imshow('line', line)
	# 	cv2.imshow('line2', luxx)
	#
	# 	_, trahs = cv2.threshold(luxx.copy(), 110, 255, cv2.THRESH_BINARY)
	#
	# 	TES(trahs)
	#
	# 	cont = cv2.findContours(trahs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	# 	cont = self.dock_sort_list(cont, typer='right_ligth')
	# 	for ind, con in enumerate(cont):
	# 		x, y, w, h = cv2.boundingRect(con)
	# 		imm = line[y:y+h, x:x+w]
	#
	# 		if imm.shape[0]<34:
	# 			word = self.TEST_read_litera(imm)
	# 			liner += word
	# 			if ind != len(cont):
	# 				liner += ' '
	# 	return liner

	def TEST_topHad_read_line(self, line):
		liner = ''
		kernel = cv2.getStructuringElement(cv2.MORPH_RECT, [10, 10])
		tophat = cv2.morphologyEx(line, cv2.MORPH_BLACKHAT, kernel=kernel, iterations=1)
		tophat= cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
		# cv2.imshow('tophat', tophat)

		luxx = cv2.erode(tophat.copy(), self.KernelUp, iterations=4)
		cv2.GaussianBlur(luxx, (3, 3), 0)


		line = self.bord_app(line, size = 20)
		luxx = self.bord_app(luxx, size = 20)
		# cv2.imshow('eroded', luxx)
		# TES(line, '|PASS|')

		cont = cv2.findContours(luxx, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		cont = self.dock_sort_list(cont, typer='right_ligth')
		for ind, con in enumerate(cont):
			x, y, w, h = cv2.boundingRect(con)
			imm = line[y:y+h, x:x+w]
			if imm.shape[0]<34:
				word = self.read_litera(imm)
				liner += word
				if ind != len(cont):
					liner += ' '


		# cv2.imshow('luxx', luxx)
		# cv2.imshow('line',line)
		# TES(tophat)


		return liner
	def standartize(self, img):
		img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=self.KernelHAT, iterations=1)
		img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)[1]
		return img
	def read_text(self, image):
		text = ''
		sizeB = 25
		image_real = image.copy()

		gray = cv2.erode(image, self.KernelXshare, iterations=3)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		_, trash = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

		"""____________________________Тест______________"""
		# TES(cv2.resize(trash, (600, 1000)))

		trash = self.bord_app(trash, size=sizeB)
		image = self.bord_app(image, size=sizeB)

		cont, _ = cv2.findContours(trash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cont = self.dock_sort_list(cont, 'Up_down')

		imgas = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
		for con in cont:
			x, y, w, h = cv2.boundingRect(con)
			imgas = cv2.rectangle(imgas, (x, y), (x+w, y+h), (0, 255, 0))

		# TES(cv2.resize(imgas, (600, 900)))
			line = image[y: y + h, x: x + w]
			if line.shape < image_real.shape:
				w1, w2 = line.shape
				if w2 > w1*2:
					line = cv2.resize(line, (line.shape[1], 17))
					if w2 > 50 :
						if w2 > 700:
							line = cv2.resize(line, (650, 17))
						line = self.TEST_topHad_read_line(line)
						text += line
						text += '\n'
						cv2.waitKey()
		return text
	def read_doc(self):
		text = ''

		image = self.image

		"""timer!"""
		# image = self.standartize(image)
		"""timer!"""

		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

		text += self.read_text(image)

		if text.replace('\n', '') == ' ':
			image = self.standartize(image)
			print("|No STANDART|")
			text += self.read_text(image)
		return text



def main():
	path1 = r"C:\Datasets\hhh.jpg"
	path2 = r"C:\Datasets\Gimage.jpeg"
	path3 = r"C:\Datasets\Pasports\4.png"
	path4 = r"C:\Datasets\wFQOjKy4Ibg.jpg"
	text = Doc_Read().read_doc()
	print(text)

if __name__ == '__main__':
	main()