import os
from tqdm import tqdm
from convert_res import get_file_basename

image_root = '/workspace/pycharm_project/mmrotate/data/2023/extra/images/'


def rename():
    files = os.listdir(image_root)
    for f in tqdm(files):
        imagePath = os.path.join(image_root, f)
        bsname = get_file_basename(imagePath)
        if 'images' in bsname:
            os.rename(imagePath, os.path.join(bsname.replace('images', '') + '.tif'))


if __name__ == "__main__":
    rename()
