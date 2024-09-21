import os
import yaml

dataset_folder = 'C:/Users/cvedi/OneDrive/Desktop/newpy/myenv/myenv/Scripts/datavals'
for f in os.listdir(dataset_folder):
    for j in os.listdir(dataset_folder+'/'+f):
        if j=='Label':
            label_files=[f for f in (os.listdir(dataset_folder+'/'+f+"/Label"))]
            break
        else:
            image_files=[f for f in (os.listdir(dataset_folder+'/'+f)) if f.endswith('.jpg')]
    print(image_files,label_files)
data_dict = {
    'train': [],
    'val': [],
    'nc': 80,  
    'names': []  
}

for class_name in os.listdir(dataset_folder):
    data_dict['names'].append(class_name)

for image_file in image_files:
    image_path = os.path.join(image_files, image_file)
    label_path = os.path.join(files, image_file.replace('.jpg', '.txt'))
    data_dict['train'].append(image_path)
    if os.path.exists(label_path):
        data_dict['val'].append(label_path)  

yaml_file_path = os.path.join(dataset_folder, 'dataset.yaml')
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(data_dict, yaml_file, default_flow_style=False)
