# How Simple Image Transformations Effect The Adversarial Nature of Images

## Description
This is Katelynn Rainey's project for the 2021 internship at the Institute for Computing in Research. Licensed under the GNU General Public License 3.0.

## Installation

To install use:
```bash
git clone https://github.com/krainey1/Convnets-and-adversarial-functions
```
then adjust file paths within .py files to test the program(s).
Also adjust image.txt file to reflect which .JPEG images will be running through the program (only applicable for adversarial_functions.py)

## Description

resnet50_test.py is an introductory script that sets up an image processing pipeline and runs it through the resnet50 convolutional neural network to print the top 5 identified class probabilities for the image input.

adversarial_fast_gradient.py
*has since been incorporated into adversarial_functions.py
Generates the adversarial image for the input image and spits back the identified class label and probability of the adversarial image and the original identification. (by changing the gradient)

adversarial_functions.py
*subject to change
This script runs 4 different methods for generating adversarial functions, which is tested by running images through an image processing pipeline and generating the class labels and probabilities for each. Output is currently in the form of a bar graph using matplotlib that autosaves graphs to the Graphs/ directory.

## Notes

*imagenet_classes.txt and and the sample images are required for running the scripts

*images.txt is required for running adversarial_functions.py (See notes in installation) as well as a Graphs/ folder.

*All other files in the repo are test results

*A lot of RAM is required to run all 100 images through adversarial_functions.py reccomended to run in batches for computers with low RAM

*adversarial_functions.py is subject to change








