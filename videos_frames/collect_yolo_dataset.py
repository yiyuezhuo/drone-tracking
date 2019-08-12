import os

path_list = []
for root, _, filename_list in os.walk('train_img'):
    for filename in filename_list:
        path = os.path.join(root, filename)
        path_list.append(path)

with open('train.txt', 'w') as f:
    f.write('\n'.join(path_list))

print("Write", len(path_list), "terms into train.txt", )

with open('test.txt', 'w') as f:
    f.write('\n'.join(path_list))

print("Create dummy test.txt")
