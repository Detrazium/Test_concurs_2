"""
Тестовое задание кандидата

Основная задача:
	Мне необходимо построить базовый алгоритм распознавания текста из документов.
	Основным документом будет являться паспорт РФ

1. 	подобрать открытые библиотеки по распознаванию документов
	и провести распознавания паспортов с указанной ссылки

2. 	подобрать API c помощью которых можно провести распознавание,
	провести распознавание

3. 	САМЫЙ ВАЖНЫЙ ПУНКТ - обучить собственный алгоритм (pytorch, tensorflow),
	которые распознает ФИО с паспорта (не пользуясь готовыми библиотеками OCR,
	но пользуясь открытыми данными по распознаванию текста)

path = r"C:\Datasets\Pasports"
||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
"""

from Create_OCR_Model import OCR_Model
from DATASETGENERATE import Start_gen
from Val_gen_split import Validat_split

def Chain_functions():
	Dataset = Start_gen()
	train, val = Validat_split(Dataset).splitD()
	history = OCR_Model(train, val).Train_model()



def main():
	Chain_functions()
if __name__ == '__main__':
	main()