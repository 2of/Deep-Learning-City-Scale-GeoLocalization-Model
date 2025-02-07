import os

'''
quic kand dirty script to replace class number in label files (i.e. RGB tragffic lights to all one class)


'''
def replace_class_number(directory, new_class_number):
    for subdir in ['train', 'test', 'valid']:
        subdir_path = os.path.join(directory, subdir)
        for root, _, files in os.walk(subdir_path):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    with open(file_path, 'w') as f:
                        for line in lines:
                            parts = line.split()
                            parts[0] = str(new_class_number)
                            f.write(' '.join(parts) + '\n')

if __name__ == "__main__":
    directory = './datasets/yolo2/onlytrafficlights'
    new_class_number = 0  # Set the new class number here
    replace_class_number(directory, new_class_number)
    
    
