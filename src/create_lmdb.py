import argparse
import pickle

import cv2
import lmdb
from path import Path

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=Path, required=True)
args = parser.parse_args()

# 2GB is enough for IAM dataset
print('open...')
assert not (args.data_dir / 'lmdb').exists()
env = lmdb.open(str(args.data_dir / 'lmdb'), map_size=1024 * 1024 * 1024 * 8)


print('go over files...')
# go over all png files
fn_imgs = list((args.data_dir / 'img').walkfiles('*.png'))
fn_imgs += list((args.data_dir / 'img').walkfiles('*.jpg'))

print(f'fn_imgs length = {len(fn_imgs)}')

print('put...')
# and put the imgs into lmdb as pickled grayscale imgs
with env.begin(write=True) as txn:
    for i, fn_img in enumerate(fn_imgs):
        print(i, len(fn_imgs))
        img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
        basename = fn_img.basename()
        txn.put(basename.encode("utf-8"), pickle.dumps(img))

env.close()
print('end')