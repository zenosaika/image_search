import imageio.v2 as imageio
import imgaug.augmenters as iaa
import os


PATH_TO_TRAIN = 'augmented_dataset/train'


class_names = [f for f in os.listdir(PATH_TO_TRAIN) if f!='.DS_Store']
for c in class_names:
    img_names = [f for f in os.listdir(f'{PATH_TO_TRAIN}/{c}') if f!='.DS_Store']
    for img in img_names:
        input_img = imageio.imread(f'{PATH_TO_TRAIN}/{c}/{img}')

        hflip= iaa.Fliplr(p=1.0)
        input_hf= hflip.augment_image(input_img)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/hflip_{img}', input_hf)

        vflip= iaa.Flipud(p=1.0) 
        input_vf= vflip.augment_image(input_img)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/vflip_{img}', input_vf)

        crop1 = iaa.Crop(percent=(0, 0.3)) 
        input_crop1 = crop1.augment_image(input_img)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/crop_{img}', input_crop1)

        noise = iaa.AdditiveGaussianNoise(10,40)
        input_noise = noise.augment_image(input_img)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/noise_{img}', input_noise)

        shear = iaa.Affine(shear=(-40,40))
        input_shear=shear.augment_image(input_img)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/shear_{img}', input_shear)

        contrast=iaa.GammaContrast((0.5, 2.0))
        contrast_sig = iaa.SigmoidContrast(gain=(5, 10), cutoff=(0.4, 0.6))
        contrast_lin = iaa.LinearContrast((0.6, 0.4))
        input_contrast = contrast.augment_image(input_img)
        sigmoid_contrast = contrast_sig.augment_image(input_img)
        linear_contrast = contrast_lin.augment_image(input_img)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/inp_contrast_{img}', input_contrast)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/sig_contrast_{img}', sigmoid_contrast)
        imageio.imsave(f'{PATH_TO_TRAIN}/{c}/lin_contrast_{img}', linear_contrast)
