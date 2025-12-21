# Deep-Learning_Mini-Project
📌 Project Title

A Multi-Task Convolutional Neural Network for Joint Age Group and Gender Prediction from Facial Images

📝 Introduction

Facial analysis is a fundamental task in computer vision with wide applications in human-computer interaction, security, demographic studies, and social media analytics. In this mini-project, we focus on building a multi-task convolutional neural network (CNN) that performs joint age group classification and gender prediction from facial images.

Instead of predicting exact ages (which can be noisy and highly variable), we classify faces into defined age groups (e.g., Child, Teen, Adult, Senior) and predict gender (Male/Female) simultaneously using a unified deep learning architecture. This approach leverages multi-task learning (MTL) to improve prediction performance by sharing features between related tasks.

The model learns shared features from input facial images through a CNN backbone and branches into two task-specific heads — one for predicting age group and another for gender.

📁 Dataset
UTKFace Dataset

We used the UTKFace dataset for training and evaluating our model. It contains over 20,000 face images with annotated age, gender, and ethnicity.

📥 Dataset source:
https://www.kaggle.com/datasets/jangedoo/utkface-new?datasetId=44109

