# Image-to-Image Search

### Never Knows Best!!
- [Link to Kaggle](https://www.kaggle.com/competitions/image-search/)
- [Link to Slide](https://drive.google.com/file/d/1pspdg44WswvPBtGxb5Lb5B5MygHxrnA7/view?fbclid=IwAR0JrKLVQMiSlolEQYzN3ZIaGVHSGKSIFFP-lHPMzTK8WMACEcH7E2IUYvY)
- [CLIP Finetuning](https://www.labellerr.com/blog/fine-tuning-clip-on-custom-dataset/)
- [CLIP Finetuning 2](https://github.com/openai/CLIP/issues/83)

### Planning
- Pre-processing (convert to grayscale, resize to the same (white padding), super resolution)
- Image Encoding -> Same Vector Space -> Similarity Search (k-nearest neighbors)
- we need to make sure that in a batch of training, no contradicting pair can be found. [Link](https://github.com/openai/CLIP/issues/83#issuecomment-1487820198)
- Image Augmentation (Flip)

### Roadmap
- Baseline CLIP + Preprocessing
- Finetuned CLIP + Preprocessing
- Preprocessing -> Feature Map from VGG19 -> k-Nearest-Neighbors