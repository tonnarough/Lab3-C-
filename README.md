Лабораторная работа #3.
====
Изучение влияние параметра “темп обучения” на процесс обучения нейронной сети на примере решения задачи классификации Oregon Wildlife с использованием техники обучения Transfer Learning
---
1)С использованием и техники обучения Transfer Learning обучить нейронную сеть EfficientNet-B0 (предварительно обученную на базе изображений imagenet) для решения задачи классификации изображений Oregon WildLife с использованием фиксированных темпов обучения 0.1, 0.01, 0.001, 0.0001
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

![Q0YuwPWV3Oc](https://user-images.githubusercontent.com/58634989/111717865-fba88100-8869-11eb-8f64-7de15df06741.jpg)


***Линейная диаграмма точности:***
<img src="./epoch_categorical_accuracy_1_part.svg">

![v5m8x1WfM4Y](https://user-images.githubusercontent.com/58634989/111717821-e7648400-8869-11eb-9f79-aab5c0035c9b.jpg)![Uploading Q0YuwPWV3Oc.jpg…]()


***Линейная диаграмма потерь:*** 
<img src="./epoch_loss_1_part.svg">  

<img src="./epoch_loss_1_part(2).svg"> 

***Анализ результатов:***



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

Также необходимо передать ***LearningRateScheduler(Планировщик скорости обучения)*** в ***callbacks(обратный вызов) - объект, который может выполнять действия на различных этапах обучения (например, в начале или в конце эпохи, до или после одной партии и т. д.).*** 
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
![ZalCT3uH9Fk](https://user-images.githubusercontent.com/58634989/111713928-96509200-8861-11eb-922e-9b1e4fae9cf4.jpg)

***Линейная диаграмма точности:***
<img src="./epoch_categorical_accuracy_2_part.svg">

![zXEYg0Zm1kQ](https://user-images.githubusercontent.com/58634989/111715627-4b387e00-8865-11eb-9e3a-d6fb1da31a34.jpg)


***Линейная диаграмма потерь:*** 
<img src="./epoch_loss_2_part.svg">  

***Анализ результатов:***



