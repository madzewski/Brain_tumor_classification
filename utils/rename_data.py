import os
path = './data/bruh/VAL'
files = os.listdir(path)


for index, file in enumerate(files):
    os.rename(os.path.join(path, file), os.path.join(path, ''.join(['y'+str(index+2100), '.jpg'])))