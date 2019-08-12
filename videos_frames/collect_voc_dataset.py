'''
Convert voc data to darknet and collect, rename them to help training calling

4_train, 5_train, 6_train, 
4_train_anno,5_train_anno,6_train_anno 
->
train_img, labels_voc
'''
import shutil
import os

root_list = ['4','5','6']
root_train_list = [root+'_train' for root in root_list]


os.makedirs('train_img', exist_ok=True)
os.makedirs('labels_voc', exist_ok=True)
os.makedirs('labels_yolo', exist_ok=True)

for root_train in root_train_list:
    root_train_anno = root_train + '_anno'
    for filename in os.listdir(root_train):
        pure_name = os.path.splitext(filename)[0]
        pure_name_tar = root_train + '_' + pure_name

        for root, root_tar, suffix in zip([root_train, root_train_anno], ['train_img', 'labels_voc'], ['.png', '.xml']):
            ori_path = os.path.join(root, pure_name + suffix)
            tar_path = os.path.join(root_tar, pure_name_tar+suffix)
            shutil.copy(ori_path, tar_path)
            print('Copy {} -> {}'.format(ori_path, tar_path))
