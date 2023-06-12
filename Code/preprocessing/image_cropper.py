from PIL import Image
import os.path, sys

path = "..\\..\\Data\\Images\\"
save_path = "..\\..\\Data\\Images_set_resized\\"
dirs = os.listdir(path)
DIM = 256
def crop():
    for item in dirs:
        f, e = os.path.splitext(item)
        if e == '.jpg':
            fullpath = os.path.join(path,item)         #corrected
            if os.path.isfile(fullpath):
                im = Image.open(fullpath)
                f, e = os.path.splitext(fullpath)
                imCrop = im.crop((96, 0, 416, 496)) #corrected
                imCrop = imCrop.resize((DIM, DIM))
                imCrop.save(save_path + item, "JPEG", quality=100)
        if e == '.png':
            fullpath = os.path.join(path,item)         #corrected
            if os.path.isfile(fullpath):
                print(item)
                im = Image.open(fullpath)
                f, e = os.path.splitext(fullpath)
                imCrop = im.crop((96, 0, 418, 500)) #corrected
                imCrop = imCrop.resize((DIM, DIM))
                imCrop.save(save_path + item, "PNG", quality=100)

crop()