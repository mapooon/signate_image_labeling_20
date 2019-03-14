# SIGNATE 画像ラベリング(20種類) Tutorial
![ranking](https://github.com/mapooon/signate_image_labeling_20/ranking_20190312.jpg)  
Competition detail:  
https://signate.jp/competitions/108
***


## Download data  
Download these files from  above URL.  
* train.zip
* test.zip
* train_master.tsv  

Unzip train.zip and test.zip.  
Place train folder:  
```
signate_image_labeling_20/data/org/train  
```
and test folder:
```  
signate_image_labeling_20/data/gen/test/test
```

## Preprocess
Run 
```
signate_image_labeling_20/src/preprocess/separate_data.py
```
For detail, please see comments in separate_data.py.

## Training
Run 
```
signate_image_labeling_20/src/models/wideresnet/train_gen.py
```
For detail, please see comments in train_gen.py.

## Predicting
Run 
```
signate_image_labeling_20/src/models/wideresnet/predict_gen.py
```
then submition.csv is generated.
For detail, please see comments in predict_gen.py.

## Submition
Submit submission.csv to above URL.
