from Dataseter import Create_dataset
from Create_OCR_Model import OCR_Model
from OCR_in_openVC import OCR_Model_find_litary
from Document_detect import Doc_Read


def Creater_Model():
	Dataset = Create_dataset().Get_data()
	train, val = Dataset
	history = OCR_Model(train, val).Train_model()


def Chain_functions():
	# path = r"C:\Datasets\Pasports\0.jpeg"
	path = r"C:\Datasets\Gimage.jpeg"
	# text = OCR_Model_find_litary(path).get_items()
	text = Doc_Read(path).read_doc()
	print(text)


def main():
	Chain_functions()

if __name__ == '__main__':
	main()