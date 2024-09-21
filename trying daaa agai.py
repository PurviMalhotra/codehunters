#creation of yaml file for yolo training
import os
import yaml
dataset_folder = 'C:/Users/cvedi/OneDrive/Desktop/newpy/myenv/myenv/Scripts/codehunters/datasetvals' 
#dic 
data_dict = {
    'train': [],
    'val': [],  
    'nc': 0,    
    'names': []  
}
for class_name in os.listdir(dataset_folder):
    class_folder = os.path.join(dataset_folder, class_name)
    if os.path.isdir(class_folder):
        data_dict['names'].append(class_name)
        data_dict['nc'] += 1  
        images_folder = os.path.join(class_folder, 'images')
        labels_folder = os.path.join(class_folder, 'labels')
        for image_file in os.listdir(images_folder):
            if image_file.endswith('.jpg'):
                image_path = os.path.join(images_folder, image_file)
                data_dict['train'].append(image_path)
                label_file = image_file.replace('.jpg', '.txt')
                label_path = os.path.join(labels_folder, label_file)
                data_dict['val'].append(label_path)
yaml_file_path = os.path.join(dataset_folder, 'datasetv2.yaml')
with open(yaml_file_path, 'w') as yaml_file:
    yaml.dump(data_dict, yaml_file, default_flow_style=False)

print(f"YAML file created at {yaml_file_path}")
