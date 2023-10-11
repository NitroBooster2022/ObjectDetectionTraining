import os
from PIL import Image
name = "images\\train"
filetype = '.jpg'
def convert_and_rename_images(folder_path):
    files = os.listdir(folder_path)
    image_files = [f for f in files if f.lower().endswith((filetype, '.png'))]
    
    for index, image_file in enumerate(image_files, start=16186):
        old_file_path = os.path.join(folder_path, image_file)
        new_file_path = os.path.join(folder_path+'2', f'0{index}{filetype}')

        # Convert PNG to JPG if necessary
        if image_file.lower().endswith('.png'):
            img = Image.open(old_file_path)
            img = img.convert('RGB')
            img.save(new_file_path, 'JPEG')
            os.remove(old_file_path)
            if index %1000 ==0:
                print(f'Converted "{image_file}" to "{index}{filetype}"')
        else:
            os.rename(old_file_path, new_file_path)
            if index %1000 ==0:
                print(f'Renamed "{image_file}" to "{index}{filetype}"')

if __name__ == '__main__':
    folder_path = "datasets\\coco128\\"+name
    if not os.path.exists(folder_path+'2'):
        os.makedirs(folder_path+'2')
    convert_and_rename_images(folder_path)
