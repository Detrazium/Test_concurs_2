import keras
import cv2
import numpy as np
from Token_word import OCR_TOKEN
import imutils
from imutils import contours


def TES(img, item=None):
	if item != None:
		print(f'|ITEM|:{item}:|')
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

		self.KernelHAT = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [20, 20])
	def bord_app(self,item, size =10):
		item = cv2.copyMakeBorder(
			item,
			top=size,
			bottom=size,
			left=size,
			right=size,
			value=(255, 255, 255),
			borderType=cv2.BORDER_CONSTANT
		)
		return item
	def find(self, image):
		litary = ''
		img = self.bord_app(image, size=20)

		img = cv2.erode(img, None, iterations=2)
		img = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)[1]
		img = cv2.GaussianBlur(img, (3, 3), 0)
		img = cv2.resize(img, (28, 28))
		num = self.Model(np.expand_dims(img, axis=0))
		num = np.argmax(num)
		litary += OCR_TOKEN(num).get_lit()
		#
		#
		# print(litary, '||', num)
		# cv2.imshow('iMG', img)
		# cv2.waitKey()
		# cv2.destroyAllWindows()
		return litary


	def strip_literaNUM(self, image):
		literas = ''
		lit_cont = []

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
		img = cv2.threshold(img, 40, 255, cv2.THRESH_BINARY)[1]

		img = cv2.erode(img, self.KernelUp, iterations=3)
		img = cv2.erode(img, self.KernelUpOne, iterations=15)

		img = self.bord_app(img, size=30)
		image = self.bord_app(image, size=30)


		cont,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cont = self.dock_sort_list(cont)

		for con in cont:

			x, y, w, h = cv2.boundingRect(con)
			img = image[y:y+h, x:x+w]
			e1, e2 = img.shape

			if 250 < e1 < 289:
				if e2 > 280:
					lit += self.strip_literaNUM(img)
				else:
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

		res1 = imutils.resize(word, height=278)

		res = cv2.GaussianBlur(res1, (3, 3), 0)
		res = cv2.dilate(res, self.KernelXshare, iterations=1)
		res = cv2.erode(res, self.KernelUpOne, iterations=20)

		res = self.bord_app(res, size=20)
		img_org = self.bord_app(res1.copy(), size= 20)

		_,Trash =cv2.threshold(res, 180, 200, cv2.THRESH_BINARY)

		cont,_ =cv2.findContours(Trash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		cont = self.dock_sort_list(cont)

		for con in cont:
			x, y, w, h =cv2.boundingRect(con)
			img = img_org[y:y+h, x:x+w]
			img = imutils.resize(img, height=205)
			w1, w2 = img.shape
			cv2.imshow('img', img)
			if 80 < w2 < 390:
				if w2 > w1*0.9:
					litera = self.Strip_one(img)

					'''CLIPED TO WORK!'''

				else:
					litera = self.find(img)
				texts += litera
		return texts

	def read_line(self, line):
		liner = ''
		luxx = cv2.erode(line.copy(), self.KernelUp, iterations=4)

		cv2.GaussianBlur(luxx, (3, 3), 0)

		line = self.bord_app(line, size = 20)
		luxx = self.bord_app(luxx, size = 20)

		_, trahs = cv2.threshold(luxx.copy(), 100, 255, cv2.THRESH_BINARY)

		cont = cv2.findContours(trahs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		cont = self.dock_sort_list(cont, typer='right_ligth')
		for ind, con in enumerate(cont):
			x, y, w, h = cv2.boundingRect(con)
			imm = line[y:y+h, x:x+w]
			if imm.shape[0]<34:
				word = self.read_litera(imm)
				liner += word
				if ind != len(cont):
					liner += ' '
		return liner
	def standartize(self, img):
		img = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel=self.KernelHAT, iterations=1)
		img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY_INV)[1]
		return img
	def read_text(self, image):
		text = ''
		gray = cv2.erode(image, self.KernelXshare, iterations=3)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		_, trash = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

		trash = self.bord_app(trash, size=25)
		image = self.bord_app(image, size=25)

		cont, _ = cv2.findContours(trash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cont = self.dock_sort_list(cont, 'Up_down')

		for con in cont:
			x, y, w, h = cv2.boundingRect(con)

			line = image[y: y + h, x: x + w]
			if line.shape != image.shape:
				w1, w2 = line.shape
				if w2 > w1*2:
					line = cv2.resize(line, (line.shape[1], 17))
					if w2 > 150 :
						if line.shape[1] > 700:
							line = cv2.resize(line, (650, 17))
						line = self.read_line(line)
						text += line
						text += '\n'
		return text
	def read_doc(self):
		text = ''

		image = self.image

		"""timer!"""
		image = self.standartize(image)
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
	text = Doc_Read(path1).read_doc()
	print(text)

if __name__ == '__main__':
	main()