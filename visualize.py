from PIL import Image
import os

def process_images(gray_folder, vi_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    gray_images = os.listdir(gray_folder)
    
    for img_name in gray_images:
        gray_img_path = os.path.join(gray_folder, img_name)
        vi_img_path = os.path.join(vi_folder, img_name)

        if os.path.exists(vi_img_path):
            gray_img = Image.open(gray_img_path).convert('L')
            vi_img = Image.open(vi_img_path).convert('YCbCr')

            _, Cb, Cr = vi_img.split()

            merged_img = Image.merge('YCbCr', (gray_img, Cb, Cr))
            
            merged_img_rgb = merged_img.convert('RGB')

            merged_img_rgb.save(os.path.join(output_folder, img_name))
            

    print("Processing completed!")

gray_folder = '' # fusion output
vi_folder = '' # input RGB image
output_folder = ''

process_images(gray_folder, vi_folder, output_folder)
