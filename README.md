Лабораторная работа #3.
====
Изучение влияния параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Oregon Wildlife с использованием техники обучения Transfer Learning
---
1)С использованием техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife с использованием фиксированных темпов обучения 0.1, 0.01, 0.001, 0.0001
---
Для этого задания был изменен темп обучения следующим образом:
```
optimizer=tf.optimizers.Adam(lr=0.1)
optimizer=tf.optimizers.Adam(lr=0.01)
optimizer=tf.optimizers.Adam(lr=0.001)
optimizer=tf.optimizers.Adam(lr=0.0001)
```
Графики обучения для нейронной сети EfficientNetB0(предварительно обученной на базе изображений imagenet) с использованием фиксированных темпов обучения 0.1, 0.01, 0.001, 0.0001:
---

***Линейная диаграмма точности:***
![XwXR0L51aqE](https://user-images.githubusercontent.com/58634989/111905068-ecd4f080-8a5a-11eb-8c3b-e1c5b9b5807f.jpg)

<img src="./epoch_categorical_accuracy_1_part.svg">

***Линейная диаграмма потерь:*** 
![zwugauPgwgM](https://user-images.githubusercontent.com/58634989/111905080-f9f1df80-8a5a-11eb-8efa-138109dafff9.jpg)

<img src="./epoch_loss_1_part.svg">  



***Анализ результатов:***
Анализируя графики можно сделать вывод, что оптимальный темп обучения 0.001, поскольку на графике метрики точности, как тренировочный, так и валидационный набор данных имеет наилучший показатель, для тренировочного 93.7%, а для валидационного 89%, по графику потерь результат для этого темпа также наилучший, минимально ошибок среди остальных, для тренировочного набора данных 0.057, а для валидационного 0.2657, однако по времени этот темп уступает всем.


2)Реализовать и применить в обучении следующие политики изменения темпа обучения, а также определить оптимальные параметры для каждой политики:
---
**a. Пошаговое затухание (Step Decay)**

**b. Экспоненциальное затухание (Exponential Decay)**

Для пошагового затухания использовалась функция:
```
def step_decay(epoch,lr):
  initial_lrate = 0.001
  drop = 0.5
  epochs_drop = 5.0
  lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
  return lrate
  ```
где:
* `initial_lrate = 0.001` - означает начальный темп обучения 
* `drop = 0.5` - снижение скорости обучение в 2 раза 
* `epochs_drop = 5.0` - каждые 5 эпох происходит снижение скорости обучения 

Для экспоненциального затухания использовалась функция:
```
def exp_decay(epoch,lr):
  initial_lrate = 0.001
  k = 0.1
  lrate = initial_lrate * math.exp(-k*epoch)
  return lrate
```

Также необходимо передать ***LearningRateScheduler(Планировщик скорости обучения)*** в ***callbacks(обратный вызов)*** - объект, который может выполнять действия на различных этапах обучения (например, в начале или в конце эпохи, до или после одной партии и т. д.).
```
callbacks=[
      tf.keras.callbacks.TensorBoard(log_dir),
      LearningRateScheduler(step_decay)
    ]
```
 Была импортирована библиотека math
 ```
 import math
 ```
Графики обучения для нейронной сети EfficientNetB0(предварительно обученной на базе изображений imagenet) с использованием следующих политик изменения темпа обучения(Пошаговое затухание (Step Decay),Экспоненциальное затухание (Exponential Decay)):
---
-Step Decay-
---
***Линейная диаграмма точности для политики изменения темпа обучения StepDecay:***
![K0QcTAZH64s](https://user-images.githubusercontent.com/58634989/111903899-5225e300-8a55-11eb-8a1a-6adf427d4a4f.jpg)

<img src="./epoch_categorical_accuracy_step.svg">

***Линейная диаграмма потерь для политики изменения темпа обучения StepDecay:*** 
![ghkIBgZn-z8](https://user-images.githubusercontent.com/58634989/111903972-944f2480-8a55-11eb-8308-072dd2da3335.jpg)

<img src="./epoch_loss_step.svg">  

***Анализ результатов:***


-Exponential Decay-
---
***Линейная диаграмма точности для политики изменения темпа обучения ExpDecay:***
![0MBxjDSbaP0](https://user-images.githubusercontent.com/58634989/111904113-37a03980-8a56-11eb-9cf3-5262fd72c1b1.jpg)


<img src="./epoch_categorical_accuracy_exp.svg">

***Линейная диаграмма потерь для политики изменения темпа обучения ExpDecay:*** 
![xeze8_4qjRc](https://user-images.githubusercontent.com/58634989/111904119-3ff87480-8a56-11eb-9b39-aea3e6044208.jpg)

<img src="./epoch_loss_exp.svg">  

***Анализ результатов:***

-Вывод-
---
***Линейная диаграмма точности для наилучших результатов всех экспериментов:***

![D6IvSuET8ZM (1)](https://user-images.githubusercontent.com/58634989/111906062-df6e3500-8a5f-11eb-93b4-29898469a800.jpg)

<img src="./epoch_categorical_accuracy_comp.svg">

***Линейная диаграмма потерь для наилучших результатов всех экспериментов:*** 

![4Twt9itmBSU](https://user-images.githubusercontent.com/58634989/111906071-e72dd980-8a5f-11eb-884f-9f33e2b3fc62.jpg)

<img src="./epoch_loss_comp.svg">  

***Вывод:***






