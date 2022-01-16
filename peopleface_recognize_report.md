---
title : 圖形識別(people face recognize by PCA and LDA)
---
[TOC]

## **使用的工具**
這次使用jupyter lab，語言為python

## **描述問題**
利用提供的人臉影像資料庫，以PCA和FLD(LDA)實作一個人臉辨識程式，並且透過分類器得出正確率和混淆矩陣


##  **實作作方法與步驟：**
#### 1.依據資料庫的影像格式，設計一個讀取pgm影像檔的函式

以下程式為將讀入的pgm檔案轉換成陣列的函式。
![](https://i.imgur.com/zqsKQXF.png)


資料庫的檔案是:s1到s40個資料夾，每個資料夾有10張照片，分別取名為1.pgm到10.pgm，所以讀取pgm影像檔時，希望能夠轉換成陣列，這樣在後續的步驟會更加容易製作。

#### 2.每張人臉影像均為92x112=10304的灰階影像，讀取後請將其轉為10304x1的向量，即成為一個樣本

#### 3.資料庫共含有400張影像（40人，每人10張），訓練時請只用200張（每人取5張）

![](https://i.imgur.com/sHXVy2d.png)

在讀入整個資料夾(人臉資料庫)後，我首先希望能夠最後分成四個部分return，分別為:train_face、train_label、test_face、test_label。

* train_face:
它是一個10304 * 200的陣列，一開始先全部填充為0，列為10304是因為每個pgm檔案在呼叫pgmtoarray()之後，會變成10304 * 1的大小，欄為200則是因為有40個人，每個人有5張照片作為訓練集。
 
* train_label:
它是一個200 * 1的陣列，專門存放每個人的標籤(s1的照片放入train_face，標籤就標記為1)。
 
* test_face:
它是一個10304 * 200的陣列，一開始先全部填充為0，和train_face相同，只是這個陣列是存放每個人有5張照片作為測試集。
 
* test_label:
和train_label相同。

train_face、train_label、test_face、test_label，四個陣列準備完畢後，再來要挑選每個人各五張照片(隨機挑選)，放入訓練集和測試集，並且更新陣列裡面的內容。

* 隨機挑選的方式:
宣告一個a，存放1到10隨機順序的陣列，然後前五個做為訓練集，後五個做為測試集。

* 放入四個陣列的方式:
挑選完畢後，train_face的0~4欄會做為s1圖片(隨機挑選)放入，下圖為s1挑選五張照片放入train_face陣列的示意圖
![](https://i.imgur.com/38dpFct.png)

更新train_label，下圖為s1挑選五張照片放入train_face陣列的示意圖
![](https://i.imgur.com/Qfn4pk0.png)

test_face和test_label也是一樣的方式存放，最後return四個陣列



#### 4.利用PCA計算此200張影像的轉換矩陣，設法將維度從10304降至10, 20, 30, 40, 50維
4.1 PCA()介紹
![](https://i.imgur.com/T7Ifkxl.png)
傳入的東西有陣列(data)和要目標維度( r )，而陣列就是前面提到的train_face，一開始先將數值改變型態為浮點數(方便後面計算)。

然後按照老師給的講義(PCA作法)
![](https://i.imgur.com/EMVNO7t.png)

* step 1:計算每一張照片的平均值data_mean，根據前面存放的方式，每一張圖片放的方式是每一欄，所以計算平均值是算每一"欄"的，然後計算得出![](https://i.imgur.com/uvCYQLW.png)(A)。
 
* step 2:得出協方差矩陣
 
* step 3:利用numpy套件的公式得出eigenvectors和eigenvalues
 
* step 4、5:排序eigenvalues然後取出前r個eigenvectors(V_r)
 
* step 6:後續將原本data轉換成PCA版本的data(final_data)
 
* step 7:return 改變後的data、data_mean、V_r

4.2 face_rec()
從前面提過的load_orl()先得出四個陣列，然後將train_face、不同維度丟入PCA()裡面，得出data_train_new,data_mean,V_r
![](https://i.imgur.com/d65PqVX.png)

然後改變test_face成為PCA的版本(降維過後的)
![](https://i.imgur.com/uyfoAlr.png)

#### 5.以這些較低維度的樣本訓練出你所學過的任何分類器來進行辨識。

然後使用KNN模型計算正確率、混淆矩陣
![](https://i.imgur.com/nRsLZTl.png)

#### 6.請比較不同維度的辨識率，並統計出混淆矩陣
以下為不同維度的各個輸出
![](https://i.imgur.com/4oteQCk.png)

![](https://i.imgur.com/WynZ4ih.png)

![](https://i.imgur.com/bTzouNM.png)

![](https://i.imgur.com/iynI7E9.png)

![](https://i.imgur.com/sWcuYzZ.png)

做成表格
![](https://i.imgur.com/V0Ho2tu.png)

#### 7.請以降維後的樣本，繼續利用FLD(LDA)找出另一轉換矩陣，利用此矩陣轉換降維後的樣本（毋需降維只須轉換）為有較佳的類別分離度之新樣本。
7.1 lda()介紹
![](https://i.imgur.com/I1zzV58.png)

根據老師的講義，有以下步驟要進行計算

![](https://i.imgur.com/UiQQY7k.png)

* step 1:要先計算Sw(Within-Class Scatter)和SB(Between-Class Scatter)。
 
* step 2:然後計算![](https://i.imgur.com/e7uqkgb.png)。
 
* step 3:利用numpy套件的公式得出eigenvectors和eigenvalues。
 
* step 4:老師題目有提到不需要再降維，所以可以直接省略取前面r個維度的部分。

* step 5:更新data成為LDA版本的新陣列並return回去，還有一個W也要return，在face_rec()的部分，需要將test的陣列根據以下公式轉換成LDA的版本。 

![](https://i.imgur.com/ozPmquD.png)

然後和PCA一樣，使用KNN模型計算正確率、混淆矩陣

![](https://i.imgur.com/Xxds6HM.png)

#### 8.以前述之辨識器再度評量辨識率以及統計混淆矩陣。

得出的辨識率、混淆矩陣和PCA版本的一樣
![](https://i.imgur.com/4oteQCk.png)

![](https://i.imgur.com/WynZ4ih.png)

![](https://i.imgur.com/bTzouNM.png)

![](https://i.imgur.com/iynI7E9.png)

![](https://i.imgur.com/sWcuYzZ.png)

做成表格
![](https://i.imgur.com/V0Ho2tu.png)

##  **結論**
根據前面表格可以發現，維度和正確率其實並沒有很明顯的相關性，每個維度的正確率大約都落在90%左右，而PCA版本的和先做過PCA在做LDA版本的結果相同。

##  **心得**
我一開始看到題目陷入一種慣性思維，就是train和test的資料要分開(資料夾)，導致前面做了新增資料夾，搬動資料等等的前置作業...，然後最開始看到pgm檔案完全不知道是甚麼東西，對於老師的1.依據資料庫的影像格式，設計一個讀取pgm影像檔的函式，也是不理解要做什麼，後來參考很多網頁的內容後才發現pgm檔案是可以用cv2套件得出陣列的，萬事起頭難，後面PCA和LDA公式雖然相似，但是做法不太一樣，所以在撰寫PCA()、lda()的時候花費了相當多的心力，在後面老師要求的6.請比較不同維度的辨識率，並統計出混淆矩陣，雖然我知道混淆矩陣是甚麼，以及要怎麼實做，但是困難點是之前我只有寫過sklearn的KNN模型，只需要fit()、predict()，最後就能得出混淆矩陣了，這次雖然也是KNN模型，作的方式完全和之前不一樣，比較偏向數學的方式計算得出結果，所以參考了以下網址進行撰寫。

https://cugtyt.github.io/blog/ml-in-action/201711081901.html
