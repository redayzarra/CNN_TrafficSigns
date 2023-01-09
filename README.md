# Traffic Signs Classification

## Overview

This project utilizes the **LeNet convolutional network architecture** to classify 43 different types of traffic signs. LeNet is a convolutional neural network that can be used for computer vision and classification models. LeNet (CNN) was designed to recognize handwritten digits. It was developed by Yann LeCun and his team at AT&T Bell Labs in the 1990s and is widely considered to be one of the first successful CNNs. LeNet consists of **several convolutional and pooling layers, followed by fully connected layers**. The architecture of LeNet has inspired many subsequent CNNs, and it is still widely used as a teaching tool for introducing the basics of CNNs to students and researchers. This project showcases a step-by-step implementation of the model and in-depth notes to customize the model further for higher accuracy. 


<div align="center">

<img src="https://user-images.githubusercontent.com/113388793/211427784-bf5e7f16-b419-45da-ac90-ca64eefdd57b.png" width="600" height="600">

</div>


## Project Website

If you would like to find out more about the project, please checkout: [Traffic Signs Classification Project](https://www.redaysblog.com/projects/traffic-signs)

## Installing the libraries

This project uses several important libraries such as Pandas, NumPy, Matplotlib, and more. You can install them all by running the following commands with pip:

```bash 
pip install pandas
pip install numpy

python -m pip install -U matplotlib
pip install seaborn

pip install -U scikit-learn
pip install tensorflow

```

If you are not able to install the necessary libraries, I recommend you **use Jupyter Notebook with Anaconda**. I have a .ipynb file for the project as well.


## Dataset and configurations

Find all the project dataset files here: [Dataset files](https://drive.google.com/drive/folders/1ctQBfS-A0YlBrbdhmH5g2993KtYyVvQU?usp=sharing)

Feel free to use your own dataset files and configure them with: 

```python
with open("YOUR-TRAINING-DATA.p", mode = 'rb') as training_data:
    train = pickle.load(training_data)

with open("YOUR-VALIDATION-DATA.p", mode = 'rb') as validation_data:
    valid = pickle.load(validation_data)

with open("YOUR-TEST-DATA.p", mode = 'rb') as testing_data:
    test = pickle.load(testing_data)
```


## License

[MIT](https://choosealicense.com/licenses/mit/)
