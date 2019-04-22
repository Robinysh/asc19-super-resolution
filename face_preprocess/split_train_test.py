import os
import numpy as np
import glob
import shutil
from progress_bar import ProgressBar
from multiprocessing import Pool

# Copy-pasting images
def worker(name,path):
    shutil.copy(name, path)

def main(data_dir=None):
    ratio = 5/1
    # # Creating Train / Val / Test folders (One time use)
    root_dir = '.'
    if data_dir == None:
        data_dir = 'input_images_align'
    n_thread = 28*8

    train_path = os.path.join(root_dir,data_dir+'_train/')
    val_path = os.path.join(root_dir,data_dir+'_val/')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    all_path = glob.glob(os.path.join(root_dir, data_dir)+'/*')

    train_amount = int(ratio/(ratio+1)*len(all_path))
    train_FileNames = all_path[:train_amount]
    val_FileNames = all_path[train_amount:]

    print('Total images: ', len(all_path))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))


    pbar = ProgressBar(len(all_path))
    def update(arg):
        pbar.update(arg)

    pool = Pool(n_thread)
    for face_img_path in train_FileNames:
        x=pool.apply_async(worker,
            args=(face_img_path,train_path),
            callback=update)
    for face_img_path in val_FileNames:
        x=pool.apply_async(worker,
            args=(face_img_path,val_path),
            callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')

if __name__ == '__main__':
    main()
