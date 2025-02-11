# Public-Health-Safety-Initiative--Mask-Detection-System
A Deep Learning model for detecting face masks in real time

## Business Scenario
In the face of global health challenges like pandemics, wearing masks has proven to be a critical measure in reducing the transmission of airborne diseases. However, ensuring widespread compliance with mask mandates remains a challenge in densely populated areas, public transportation systems, and workplace environments.

Imagine a scenario where a state-of-the-art AI system automatically monitors and identifies individuals not wearing masks in real-time. This technology can enhance public safety measures, streamline enforcement efforts, and contribute to healthier communities. By leveraging advanced machine learning techniques and robust neural networks, this project aims to develop an automated solution for mask detection using computer vision.

This mask detection system can be seamlessly integrated into surveillance systems, helping authorities and organizations enforce safety protocols efficiently. It can be deployed in high-traffic areas like airports, malls, schools, and offices to ensure compliance, minimize health risks, and maintain a safe environment for everyone.

By utilizing transfer learning with powerful pre-trained models and TensorFlow's capabilities, this project seeks to develop a reliable and efficient mask detection system. This AI-powered solution offers significant advantages, including:

- Enhancing public safety: Ensures compliance with mask mandates, reducing the risk of airborne disease transmission.
- Streamlining enforcement efforts: Provides real-time alerts for non-compliance, enabling prompt corrective actions.
- Improving operational efficiency: Reduces the need for manual monitoring, allowing human resources to focus on more critical tasks.

## Dataset
There are 4 dataset that are used in this project , three of them used in training, which are :
- MaskNet dataset by Adnane Cabani , Karim Hammoudi, Halim Benhabiles , Mahmoud Melkemi and Junhao Cao [(dataset link)](https://github.com/cabani/MaskedFace-Net?tab=readme-ov-file)
- Face mask lite dataset by prasoon kottarathil [(dataset link)](https://www.kaggle.com/datasets/prasoonkottarathil/face-mask-lite-dataset/)
- Real time face mask detection dataset by Piyush2912 [(dataset link)](https://github.com/Piyush2912/Real-Time-Face-Mask-Detection?tab=readme-ov-file)

for validation and testing , the dataset used is :
- Face Mask Detection by Ashish Jangra [(dataset link)](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

## Model Architecture
The architecture used for this project is Resnet50V2 , with the following custom layers on top of ResNet50:
- Average pooling layer
- dense layer with relu activation function (128 neuron)
- dropout layer for regularization (set to 0.5)
- output denselayer with sigmoid activation function (for binary classification purpose)


## Model compilation 
The model is compiled using the following setup:
- optimizer : Adam optimizer(with learning rate set to 0.001)
- loss function: binary CrossEntropy.
- metric: accuracy

## Model Training
The model was trained using the following setup:
- Early stopping applied with patience set to 5
- epochs : 40

## Results
after 14 epochs the final model achieved the following performance:

- Train Accuracy: 99.93% | Train Loss : 0.0026
- Validation Accuracy: 96.19% | Validation Loss : 0.1490
- Test Accuracy : 97.75% | Test Loss : 0.0676


## Deployment
Model is Depoloyed on hugging face , check it [here](https://huggingface.co/spaces/Mamdouh-Alaa12/Public-Health-Safety-Initiative-Mask-Detection-System)

## Notes
- The model was tested locally in real time , you will find the code in model_realtime_test.py , but when deployed we couldn't use real time detection with on platforms like hugging faces ,due to security reasons , so in the deployment environment on hugging face ,  we proceeded with video upload instead of real time detection, the code is found in app.py.
- when tested locally , we used OpenCv DNN for face detection , which is more accurate than haarcascade , but in the deployed environment , we proceeded with haarcascade for face detection , since it's more ligh weight to work on a cloud platform like hugging face (free plan without GPU).
