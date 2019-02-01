import os
from extract_subimgs_single import main as extract_subimg
from create_lmdb import main as create_lmdb 
import subprocess

base_dir = '/data/data/superresolution/'
input_dir = 'DIV2K_valid_HR'

output_dir = base_dir+input_dir+'_sub'
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
