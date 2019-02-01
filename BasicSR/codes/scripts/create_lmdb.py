import sys
import os.path
import glob
import pickle
import lmdb
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.progress_bar import ProgressBar

def main(img_folder=None, lmdb_save_path=None):
    # configurations
    #img_folder = 'data/DIV2K_train_HR/*'  # glob matching pattern
    #lmdb_save_path = 'data/DIV2K_train_HR.lmdb'  # must end with .lmdb
    if  img_folder is None:
        img_folder = '/home/robinysh/media/usb/lustre/scratch/asc19/superresolution/dataset/DIV2K_valid_HR_sub_bicLRx4/*'  # glob matching pattern
    if  lmdb_save_path is None:
        lmdb_save_path = '/home/robinysh/media/usb/lustre/scratch/asc19/superresolution/dataset/DIV2K_valid_HR_sub_bicLRx4.lmdb'  # glob matching pattern
    #img_folder = '/home/robinysh/media/usb/lustre/scratch/asc19/superresolution/dataset/DIV2K_train_HR_sub/*'  # glob matching pattern
    #lmdb_save_path = '/home/robinysh/media/usb/lustre/scratch/asc19/superresolution/dataset/DIV2K_train_HR_sub.lmdb'  # glob matching pattern

    img_list = sorted(glob.glob(img_folder))
    dataset = []
    data_size = 0

    print('Read images...')
    pbar = ProgressBar(len(img_list))
    for i, v in enumerate(img_list):
        pbar.update('Read {}'.format(v))
        img = cv2.imread(v, cv2.IMREAD_UNCHANGED)
        dataset.append(img)
        data_size += img.nbytes
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    print('Finish reading {} images.\nWrite lmdb...'.format(len(img_list)))

    pbar = ProgressBar(len(img_list))
    with env.begin(write=True) as txn:  # txn is a Transaction object
        for i, v in enumerate(img_list):
            pbar.update('Write {}'.format(v))
            base_name = os.path.splitext(os.path.basename(v))[0]
            key = base_name.encode('ascii')
            data = dataset[i]
            if dataset[i].ndim == 2:
                H, W = dataset[i].shape
                C = 1
            else:
                H, W, C = dataset[i].shape
            meta_key = (base_name + '.meta').encode('ascii')
            meta = '{:d}, {:d}, {:d}'.format(H, W, C)
            # The encode is only essential in Python 3
            txn.put(key, data)
            txn.put(meta_key, meta.encode('ascii'))
    print('Finish writing lmdb.')

    # create keys cache
    keys_cache_file = os.path.join(lmdb_save_path, '_keys_cache.p')
    env = lmdb.open(lmdb_save_path, readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin(write=False) as txn:
        print('Create lmdb keys cache: {}'.format(keys_cache_file))
        keys = [key.decode('ascii') for key, _ in txn.cursor()]
        pickle.dump(keys, open(keys_cache_file, "wb"))
    print('Finish creating lmdb keys cache.')

if __name__ == '__main__':
    main()
