import os

root_folder_train = '/data/disk2/vinhnguyen/Dino/test'
root_folder_test = '/data/disk2/vinhnguyen/Dino/valid'

check_train = []
check_test = []

for folder in os.listdir(root_folder_train):
    folder_path = os.path.join(root_folder_train, folder)
    check_train.append(len(os.listdir(folder_path)))
    
# for folder in os.listdir(root_folder_test):
#     folder_path = os.path.join(root_folder_test, folder)
#     check_test.append(len(os.listdir(folder_path)))
            
# check_all = [int(check_train[x] / check_test[x]) for x in range(0, len(check_train))]
print(check_train)
print(check_test)
# print(check_all)       

print(sum(check_train))