import keras
import cv2
import numpy as np
from Token_word import OCR_TOKEN
import imutils
from imutils import contours


def TES(img):
	cv2.imshow(f'{img.shape}|TEST READ: ', img)
	cv2.waitKey()
	cv2.destroyAllWindows()

class Doc_Read():
	def __init__(self, doc=None):
		self.Doc = doc
		self.Model = keras.models.load_model(r'C:\Users\Антонио\PycharmProjects\Test_concurs\OCR_models\test_model_ocr_recovVV3.h5')
		self.KernelXshare = cv2.getStructuringElement(cv2.MORPH_RECT, [10, 2])
		self.KernelUp = cv2.getStructuringElement(cv2.MORPH_RECT, [4, 10])

	def find(self, image):
		img = self.bord_app(image, size=30)
		# img = cv2.erode(img, None, iterations=4)

		img = cv2.resize(img, (28, 28))
		num = self.Model(np.expand_dims(img, axis=0))
		num = np.argmax(num)
		litary = OCR_TOKEN(num).get_lit()


		print(litary, '||', num)
		cv2.imshow('iMG', img)
		cv2.waitKey()
		cv2.destroyAllWindows()
		return litary
	def dock_sort_list(self, cnt, typer='right_ligth'):
		rev = False
		if typer == 'right_ligth':
			typer = 0
		if typer == 'Up_down':
			typer = 1

		Bound = [cv2.boundingRect(p) for p in cnt]
		(cnt, Bound) = zip(*sorted(zip(cnt, Bound), key = lambda b: b[1][typer], reverse=rev))
		return cnt
	def bord_app(self,item, size =9):
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

	def read_litera(self, word):
		res = imutils.resize(word, height=278)
		img_org = res.copy()
		cv2.GaussianBlur(res, (3, 3), 0)
		res = cv2.erode(res, self.KernelUp, iterations=1)


		# image_orig = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
		_,Trash =cv2.threshold(res, 100, 200, cv2.THRESH_BINARY)

		Trash = cv2.erode(Trash, self.KernelUp, iterations=2)

		cont,_ =cv2.findContours(Trash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		# cv2.drawContours(image_orig, cont, -1, (0, 255, 0))
		cont = self.dock_sort_list(cont)

		texts = ''
		for con in cont:
			x, y, w, h =cv2.boundingRect(con)
			img = img_org[y:y+h, x:x+w]
			w1, w2 = img.shape
			if 140<w1<190 and 110 < w2:
				litera = self.find(img)
				texts += litera
		return texts

	def read_line(self, line):
		liner = ''
		luxx = cv2.erode(line.copy(), self.KernelUp, iterations=2)

		cv2.GaussianBlur(luxx, (3, 3), 0)
		line = self.bord_app(line)
		luxx = self.bord_app(luxx)
		_, trahs = cv2.threshold(luxx.copy(), 100, 255, cv2.THRESH_BINARY)

		cont = cv2.findContours(trahs, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
		cont = self.dock_sort_list(cont, typer='right_ligth')
		for ind, con in enumerate(cont):
			x, y, w, h = cv2.boundingRect(con)
			img = cv2.rectangle(luxx.copy(), (x, y), (y+h, x+w), (0, 0, 255))
			imm = line[y:y+h, x:x+w]
			if imm.shape[0]<34:
				word = self.read_litera(imm)
				liner += word
				if ind != len(cont):
					liner += ' '
		return liner
	def read_doc(self):
		path = self.Doc
		image = cv2.imread(path)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		gray = cv2.erode(image, self.KernelXshare, iterations=3)
		gray = cv2.GaussianBlur(gray, (3, 3), 0)
		_, trash = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
		cont, _ = cv2.findContours(trash, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cont = self.dock_sort_list(cont, 'Up_down')

		text = ''
		for con in cont:
			x, y, w, h = cv2.boundingRect(con)
			line = image[y: y+h, x: x+w]
			if line.shape[0] < 40:
				line = self.read_line(line)
				text += line
				text += '\n'
		return text



def main():
	text = Doc_Read(r"C:\Datasets\IMAGE.jpeg").read_doc()
	print(text)

if __name__ == '__main__':
	main()