from Dataseter import Create_dataset
from Create_OCR_Model import OCR_Model
from OCR_in_openVC import OCR_Model_find_litary


def Creater_Model():
	Dataset = Create_dataset().Get_data()
	train, val = Dataset
	history = OCR_Model(train, val).Train_model()

def Chain_functions():
	path = r"C:\Datasets\Pasports\0.jpeg"
	text = OCR_Model_find_litary(path).get_items()
	print(text)


def main():
	Chain_functions()

if __name__ == '__main__':
	main()