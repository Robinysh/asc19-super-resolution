import os
from generate_datafile_scripts.extract_subimgs_single import main as extract_subimg
from generate_datafile_scripts.create_lmdb import main as create_lmdb 
from face_alignment import main as face_alignment 
from split_train_test import main as split_train_test 
import subprocess

base_dir = './'
input_dir = 'input_images'

input_dir = base_dir+input_dir
output_dir = input_dir+'_align'
if not os.path.isdir(output_dir):
    face_alignment(base_dir+input_dir+'/*', output_dir, 'shape_predictor_5_face_landmarks.dat')
else:
    print('align file already exist')

input_dir = output_dir
output_dir = input_dir+'_val'
if not os.path.isdir(output_dir):
    split_train_test(input_dir)
else:
    print('train val file already exist')


#output_dir = output_dir+'_sub'
'''
if not os.path.isdir(output_dir):
    extract_subimg(base_dir+input_dir+'/', output_dir)
else:
    print('sub file already exist')

input_dir = output_dir
output_dir = input_dir+'.lmdb'
if not os.path.isdir(output_dir):
    create_lmdb(input_dir+'/*', output_dir)
else:
    print('sub lmdb file already exist')
'''
old_input_dir = input_dir
for pre in ['_train', '_val']:
    input_dir = old_input_dir+pre
    output_dir = input_dir+'.lmdb'
    if not os.path.isdir(output_dir):
        create_lmdb(input_dir+'/*', output_dir)
    else:

        print('lmdb file already exist')

    output_dir = input_dir+'_bicLRx4'
    if not os.path.isdir(output_dir):
        subprocess.run("matlab -nodesktop -nosplash -r \"generate_mod_LR_bic(\'{}\',\'{}\'); exit;\"".format(input_dir, output_dir), shell=True)
    else:
        print('sub LRx4 file already exist')

    input_dir = output_dir
    output_dir = input_dir+'.lmdb'
    if not os.path.isdir(output_dir):
        create_lmdb(input_dir+'/*', output_dir)
    else:
        print('sub LRx4 lmdb file already exist')
