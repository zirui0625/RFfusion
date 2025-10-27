import os
def get_vi_image_paths(root_dir, output_train_file, output_test_file):
    with open(output_train_file, 'w') as train_f, open(output_test_file, 'w') as test_f:
        for dataset in os.listdir(root_dir):
            dataset_dir = os.path.join(root_dir, dataset)
            if os.path.isdir(dataset_dir):
                vi_dir = os.path.join(dataset_dir, 'vi')
                if os.path.isdir(vi_dir):
                    for sub_dir in ['train', 'test']:
                        sub_dir_path = os.path.join(vi_dir, sub_dir)
                        if os.path.isdir(sub_dir_path):
                            for root, dirs, files in os.walk(sub_dir_path):
                                for file in files:
                                    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                                        file_path = os.path.join(root, file)
                                        if sub_dir == 'train':
                                            train_f.write(file_path + '\n')
                                            print(f"Found train image: {file_path}")
                                        elif sub_dir == 'test':
                                            test_f.write(file_path + '\n')
                                            print(f"Found test image: {file_path}")

if __name__ == "__main__":
    root_dir = 'data'  
    output_train_file = 'data/vi_train_image_paths.txt'  
    output_test_file = 'data/vi_test_image_paths.txt'   

    get_vi_image_paths(root_dir, output_train_file, output_test_file)
    print(f"All train image paths have been saved to {output_train_file}")
    print(f"All test image paths have been saved to {output_test_file}")
