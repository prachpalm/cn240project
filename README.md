# CN240 : Data Science for Signal Processing
## Project : Glaucoma Classification

```
1. Chonlasit Mooncorn 6110613020
2. Prach Chantasantitam 6110613202
3. Weeraphat Leelawittayanon 6210612823
4. Raned Chuphueak 6210612864
```
ไฟล์ประกอบไปด้วย 2 โฟลเดอร์ ซึ่งเป็น code สำหรับการพัฒนา ได้แก่ โฟลเดอร์ DL (Deep Learning) และโฟลเดอร์ ML (Machine Learning) สำหรับใช้ในการจำแนกภาพจอประสาทตา (Retina Fundus Image) ว่าเป็น Glaucoma หรือไม่

Folder: DL
-Folder DenseNet121 เป็นโฟลเดอร์ที่ใช้สำหรับเก็บโมเดล DenseNet121 ที่ใช้สามารถใช้ train และสามารถให้ output ออกมาเป็น Graph และ  Matrix ได้
-Folder VGG16 เป็นโฟลเดอร์ที่ใช้สำหรับเก็บโมเดล VGG16 ที่สามารถใช้ train และสามารถให้ output ออกมาเป็น Graph และ Matrix ได้
-Folder gsaved_models เป็นไฟล์ที่ไว้ใช้สำหรับเก็บ model glaucoma ที่ดีที่สุด เพื่อไว้ใช้ deploy ใน api-dl.py ไฟล์ด้านล่าง
-Folder nsaved_models เป็นไฟล์ที่ไว้ใช้สำหรับเก็บ model normal ที่ดีที่สุด เพื่อไว้ใช้ deploy ใน api-dl.py ไฟล์ด้านล่าง
-Folder osaved_models เป็นไฟล์ที่ไว้ใช้สำหรับเก็บ model other ที่ดีที่สุด เพื่อไว้ใช้ deploy ใน api-dl.py ไฟล์ด้านล่าง
-api-dl.py เป็นไฟล์ที่ใช้สำหรับการ deploy model deep learning เพื่อใช้ในการยิงทดสอบ
-dataset_test.csv เป็นไฟล์ที่บอกชื่อรูป และ label status ไว้สำหรับ train model (0,1,2 คือ glaucoma, normal, other ตามลำดับ)
-dataset_train.csv เป็นไฟล์ที่บอกชื่อรูป และ label status แยกไว้สำหรับ test model หลังจากที่ train model เสร็จเรียบร้อยแล้ว
-datasetextractor.py เป็นไฟล์สำหรับใช้ในการอ่านชื่อรูปทั้งหมดในโฟลเดอร์ และแตกไฟล์ออกมาเป็น csv เหมือนกับ dataset_test และ dataset_train แต่เวลาใช้ต้องปรับค่าชื่อโฟลเดอร์ และชื่อไฟล์กับโค้ดบางส่วนเสียก่อน

Folder: ML
-preprocessing.ipnyb คือ ตัว pre processing ของ machine learning มีไว้เพื่อ run ภาพใน dataset เพื่อเก็บข้อมูลตัวแปรต่างๆ และออกมาเป็นไฟล์ dataset.csv เพื่อใช้ในการประมวลผลต่อไป
-dataset.csv เป็นไฟล์ที่เก็บข้อมูลหลังจากการผ่านขั้นตอน pre processing เพื่อนำไปใช้ในการประมวลผลและวาดกราฟต่อไป
-Glaucoma.ipynb เป็นไฟล์ที่ใช้สำหรับจัดทำ model glaucoma ของตัว machine learning โดยผ่านวิธีการต่างๆ ทั้ง SVM(Support Vector Machine) KNN (K-nearest neighbors) LG (Linear Regression) NB (Naieve Bayes) โดยตัวโมเดลจะไปเซฟไว้ในโฟลเดอร์ใหม่ที่ที่ผู้ใช้ต้องตั้งชื่อว่า Model/glaucoma
-Normal.ipynb  เป็นไฟล์ที่ใช้สำหรับจัดทำ model normal ของตัว machine learning โดยผ่านวิธีการต่างๆ ทั้ง SVM(Support Vector Machine) KNN (K-nearest neighbors) LG (Linear Regression) NB (Naieve Bayes) โดยตัวโมเดลจะไปเซฟไว้ในโฟลเดอร์ใหม่ที่ผู้ใช้ต้องตั้งชื่อว่า Model/normal
-Other.ipynb  เป็นไฟล์ที่ใช้สำหรับจัดทำ model other ของตัว machine learning โดยผ่านวิธีการต่างๆ ทั้ง SVM(Support Vector Machine) KNN (K-nearest neighbors) LG (Linear Regression) NB (Naieve Bayes) โดยตัวโมเดลจะไปเซฟไว้ในโฟลเดอร์ใหม่ที่ผู้ใช้ต้องตั้งชื่อว่า Model/other
-Folder Graph จัดเก็บ output เป็นกราฟ ROC ของไฟล์ Glaucoma Normal Other.ipnyb
-Folder Matrix จัดเก็บ output เป็นกราฟ confusion matrix ของไฟล์ Glaucoma Normal Other.ipnyb
