As part of the project, we are planning to attack a multimodal classification model on the Hateful Memes dataset. You can download the dataset from this file. The objective of the project is to minimize the classification accuracy by adversarially modifying the image and/or text of memes to maximise the misclassification of memes from hateful to non-hateful and vice-versa. Using the adversarially modified images and/or texts.

To run the code on a conda environment, you need to create an environment with packages installed from requirements.txt
1. Create a conda environment conda env create --file requirement.txt
2. To run image perturbations, run the code in the img_perturbation folder.
3. To run text perturbations, run the code in the text_perturbation folder.
4. To run text and image perturbations, run the code in the text_perturbation folder.
5. Adversarial retraining can be done through the code in adv_retrain folder.


To run a attack in any of the folders above, you can use the command python attack.py

Make sure you download the perturbed images from here and put in the img_perturbation folder before running a attack.
