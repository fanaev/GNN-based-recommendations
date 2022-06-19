# Курсовая работа на тему "Построение рекомендательной системы с помощью графовой нейронной сети"
**В работе были использованы:**
1) Данные из соревнования от H&M https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/overview
2) модель Pinsage https://arxiv.org/pdf/1806.01973.pdf
3) Реализация модели от разработчтков dgl https://github.com/dmlc/dgl/tree/master/examples/pytorch/pinsage.

**Результаты:**

  Модель обучалась 41 эпоху, итоговое значение функции потерь: 0.3630964728664607
  
<img width="482" alt="image" src="https://user-images.githubusercontent.com/81157248/174504226-3511d12e-0ff8-48ee-9adf-3fb09e7a9ba1.png">

**Пример:**

Исследуемый товар

<img width="157" alt="image" src="https://user-images.githubusercontent.com/81157248/174504311-0de2361d-50e7-4a58-a67a-67c8611eb0f6.png">

Рекоммендации / топ 3 схожих товаров

<img width="482" alt="image" src="https://user-images.githubusercontent.com/81157248/174504410-4a697be5-fed8-4030-aac5-1521ef154266.png">

