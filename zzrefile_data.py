import os
from PIL import Image

def rename_images(root_folder):
    for class_folder in os.listdir(root_folder):
        class_path = os.path.join(root_folder, class_folder)
        if os.path.isdir(class_path):
            class_id = 1
            image_files = [filename for filename in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, filename)) and any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif'])]
            for filename in image_files:
                file_path = os.path.join(class_path, filename)
                try:
                    img = Image.open(file_path)
                    img.load()
                    img.close()
                except (IOError, SyntaxError) as e:
                    print(f"Deleting {file_path} due to error: {e}")
                    os.remove(file_path)
                    continue

                new_filename = f"test_{class_folder}_{class_id}.jpg"
                new_file_path = os.path.join(class_path, new_filename)
                os.rename(file_path, new_file_path)
                print(f"Renamed {file_path} to {new_file_path}")
                class_id += 1


root_directory = '/data/disk2/vinhnguyen/Dino/test'
rename_images(root_directory)
