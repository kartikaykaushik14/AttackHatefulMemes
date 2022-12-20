As part of the project, we are planning to attack a multimodal classification model on the Hateful Memes dataset. You can download the dataset by running code download_dataset.py . The objective of the project is to minimize the classification accuracy by adversarially modifying the image and/or text of memes to maximise the misclassification of memes from hateful to non-hateful and vice-versa. Using the adversarially modified images and/or texts.

To run the code on a conda environment, you need to create an environment with packages installed from requirements.txt
1. Create a conda environment conda env create --file requirement.txt
2. To run image perturbations, run the code in the img_perturbation folder.
3. To run text perturbations, run the code in the text_perturbation folder.
4. To run text and image perturbations, run the code in the text_perturbation folder.
5. Adversarial retraining can be done through the code in adv_retrain folder.


To run a attack in any of the folders above, you can use the command python attack.py

Make sure you download the perturbed images from [here](https://drive.google.com/drive/folders/19Cgq2q-csOgrsa0bAUcaeEOYvlekKzM2) and put in the folder called annotations inside the original dataset.

Make sure you have the correct path in line [63](https://github.com/kartikaykaushik14/AttackHatefulMemes/blob/main/img_perturbation/attack.py#L63) and [67](https://github.com/kartikaykaushik14/AttackHatefulMemes/blob/main/img_perturbation/attack.py#L67) in all the attack.py files.
