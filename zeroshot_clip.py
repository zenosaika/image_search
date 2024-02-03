from transformers import (
    AutoModel,
    CLIPProcessor
)
import os
import pandas as pd
from PIL import Image
import torch


if __name__ == '__main__':
    src_dir = 'test/images'
    query_dir = 'queries/queries'
    submission = pd.read_csv('sample_submission.csv')
    model = AutoModel.from_pretrained('openai/clip-vit-base-patch32').to("mps").eval()
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

    submission['dot_class'] = 22
    submission['cosine_class'] = 22
    with torch.no_grad():
        query_images = []
        query_classes = []
        for file in os.listdir(query_dir):
            inputs = processor(images=[Image.open(os.path.join(query_dir, file)).convert('RGB')], return_tensors='pt').to('mps')
            outputs = model.get_image_features(inputs.pixel_values).cpu()
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            query_images.append(outputs)
            query_classes.append(int(file[:-4]))
        query_images = torch.cat(query_images)
        for idx, row in submission.iterrows():
            if not pd.isna(row['class']):
                continue
            inputs = processor(images=[Image.open(os.path.join(src_dir, row['img_file'])).convert('RGB')], return_tensors='pt').to('mps')
            outputs = model.get_image_features(inputs.pixel_values).cpu()
            outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
            values = outputs @ query_images.T
            if values.softmax(1).max() > .055:
                submission.at[idx, 'dot_class'] = query_classes[values.argmax().numpy().tolist()]
            cosine = torch.cosine_similarity(outputs, query_images)
            if cosine.max() > 0.8:
                submission.at[idx, 'cosine_class'] = query_classes[cosine.argmax().numpy().tolist()]

        sub = submission[['img_file',]]
        sub['class'] = submission['dot_class']
        sub.to_csv('dot_product.csv', index=False)
        sub['class'] = submission['cosine_class']
        sub.to_csv('cosine_similarity.csv', index=False)
