# import os
# from PIL import Image

# # Define the path to the dataset folder
# dataset_folder = 'archive'

# # Define the folder containing the happy images
# happy_folder = os.path.join(dataset_folder, 'train', 'happy')  # Adjust 'train' to 'test' if needed

# # List all image files in the happy folder
# happy_images = [os.path.join(happy_folder, f) for f in os.listdir(happy_folder) if os.path.isfile(os.path.join(happy_folder, f))]

# # Load and display the first happy image
# if happy_images:
#     first_happy_image = Image.open(happy_images[0])
#     first_happy_image.show()
# else:
#     print("No happy images found in the dataset.")
import os
from PIL import Image

# Define the path to the dataset folder
dataset_folder = 'archive'

# Define the folder containing the sad images
sad_folder = os.path.join(dataset_folder, 'train', 'sad')  # Adjust 'train' to 'test' if needed

# List all image files in the sad folder
sad_images = [os.path.join(sad_folder, f) for f in os.listdir(sad_folder) if os.path.isfile(os.path.join(sad_folder, f))]

# Load and display the first sad image
if sad_images:
    first_sad_image = Image.open(sad_images[0])
    first_sad_image.show()
else:
    print("No sad images found in the dataset.")
