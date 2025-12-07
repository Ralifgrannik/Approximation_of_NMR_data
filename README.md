---
Приложение для аппроксимации релаксационных затуханий

<img width="1276" height="912" alt="Frame 2 (1)" src="https://github.com/user-attachments/assets/7fbaeff5-03ee-49c6-8fcf-588bfcd1b501" />

---
# Approximation-of-NMR-data
Programs for approximating relaxation attenuations (determination of components and their relaxation times T2 using neural networks). The model recognizes up to 4 components. Training takes place for T2 times from 0.1s to 5s. The minimum possible component is 0.1.
In the repository:
- Model_training.ipynb (code for training the model)
- Sample_data.ipynb (code for viewing an example of generated data)
- Model_test.ipynb (Code for checking the model on test data)
- A file with model weights
- All.ipynb (a shared file with all code blocks for ease of use)
- NMR_T2_Analyzer.py (the code for creating the application .exe)

Программы для аппроксимации релаксационных затуханий(определение компонент и их времен релаксации Т2 с помощью нейросетей). Модель распознает до 4 компонент. Обучение идет для времён Т2 от 0.1с до 5с. Минимально возможная компонента - 0.1.
В репозитории:
- Model_training.ipynb (код для обучения модели)
- Sample_data.ipynb (код для просмотра примера генерируемых данных)
- Model_test.ipynb (Код для проверки модели на тестовых данных)
- Файл с весами модели
- All.ipynb (общий файл со всеми блоками кода для удобства работы)
- NMR_T2_Analyzer.py (код, для создания приложеня .exe)
---

## Инструкция по скачиванию приложения
Link to the application(Ссылка на приложение):
https://drive.google.com/drive/folders/1OEft7p4sze9gBZsQerY5Q2LAXrNj-Bdp?usp=sharing

## Инструкция:
Разархивируйте архив в любую удобную для вас директорию. В папке вы найдете папку dist, в которой будет приложение .exe. Ни в коем случае не вытаскивайте его оттуда. Нажмите на приложение правой кнопкой мыши, копируйте его и вставьте ярлык на рабочий стол. 
В случае, если вы обучили модель в Google Colab и получили лучшие значения при аппроксимации, чем в данной версии приложения, то можете заменить файл best_nmr_param_cnn.pth на свою версию.

Instructions:
Unzip the archive to any directory convenient for you. In the folder you will find the dist folder, which will contain the .exe application. In any case, do not pull it out of there. Right-click on the application, copy it and paste the shortcut to the desktop. 
If you trained the model in Google Colab and got better approximation values than in this version of the application, you can replace the best_nmr_param_cnn.pth file with your own version.

---
