# Image Search (Logo Classification)

### How to Train Your Dragon (AI)
- Install Requirements using `pip install -r requirements`
- Run `img_augment.py` to augment images in `augmented_dataset`
- Run `finetune.py` to finetune YOLOv8 with our dataset
- Run `inference.py` to predict class of the testing set and export to csv

### Review
- ก่อนแข่ง
    - ไปกินบุปเฟ่ต์เพื่อเรียกขวัญกำลังใจกับเพื่อน
    - ยืนโหนราวฟังโจทย์ใน Zoom ระหว่างนั่งรถเมย์กลับมหาลัย
    - โต้รุ่ง ณ ห้องสมุดมหาลัย TT
- ระหว่างแข่ง
    - ในวันแรกเราใช้ CLIP ในการทำ Image to Image Search -> Score = 0.610
    - ในวันที่สอง เราลองเปลี่ยนมาใช้ YOLOv8 โดยทำ Image Augmentation (Flip, Crop, Noise, Contrast) จากนั้นนำไปเทรนบน Vast.ai (GPU Rental) -> Score = 0.979
    - เรามีการกรอง training set ก่อนเอามาเทรน เช่น HP กับ Hewlett Packard มันก็อันเดียวกัน แต่ดันแยกอยู่ 2 โฟลเดอร์ (กลายเป็น 2 class นำไปสู่ conflict) เราก็จับมารวมกัน
- หลังแข่งจบ
    - ไปเดินงาน Japan Expo @Central World เห็นโลโก้แต่ละร้านแล้วหลอนมากครับ
    - หลังจากอ่านโค้ดคนอื่นคร่าว ๆ รู้สึกเปิดโลกสุด ๆ แบบว่ามีการทำ ensemble เอา model หลาย ๆ ตัวมาช่วยกันตัดสินใจ กับทำ OCR เพื่อ post processing พวก readable text ใน logo ซึ่งทำให้แยกพวกโลโก้ Louis Vuitton กับ Saint Lawrent อะไรประมาณนี้ได้แม่นขึ้น (ส่วนเราแค่ทำ data augmentation แล้วก็เทรนยาวไป รู้สึกธรรมดามาก Orz)
    - โค้ดของเรา ถ้าเราลองปรับจูน confidence level ดู ผลลัพธ์อาจจะดีขึ้นก็ได้ (แต่ยังไม่ได้ลองทำ)

### Resources
- [Link to Kaggle](https://www.kaggle.com/competitions/image-search/)
- [Link to Slide](https://drive.google.com/file/d/1pspdg44WswvPBtGxb5Lb5B5MygHxrnA7/view?fbclid=IwAR0JrKLVQMiSlolEQYzN3ZIaGVHSGKSIFFP-lHPMzTK8WMACEcH7E2IUYvY)
- [CLIP Finetuning](https://www.labellerr.com/blog/fine-tuning-clip-on-custom-dataset/)
- [CLIP Finetuning 2](https://github.com/openai/CLIP/issues/83)
