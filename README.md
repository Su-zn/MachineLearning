📌 Project Overview
This repository contains a complete deep learning pipeline built with TensorFlow and Keras. The project benchmarks multiple neural network architectures on a fast-food dataset to optimize image classification performance on a GPU setup.

The workflow includes automated corrupt image purging, high-throughput TensorFlow data pipelines, mixed-precision training, callbacks, and comparative evaluation metrics.

🛠️ Key Features
Automated Data Cleaning: Audits raw images using PIL and imghdr to detect and remove corrupt files prior to training.

Optimized Data Pipeline: Leverages TensorFlow image_dataset_from_directory with AUTOTUNE prefetching, caching, and batching.

Multi-Model Benchmarking: Iterates through 3 distinct architectures:

Baseline CNN: Built from scratch with Conv2D, MaxPooling, and Dropout layers.

Transfer Learning Model: Leverages a pre-trained backbone (e.g., MobileNetV2 / ResNet) with frozen weights for fast feature extraction.

Fine-Tuned Model: Unfreezes top layers of the pre-trained backbone with custom dense layers for specialized accuracy.

Advanced Training Workflows: Integrates EarlyStopping, ReduceLROnPlateau, and mixed-precision policies for efficient model convergence.

Model Evaluation & Visualization: Generates confusion matrices, ROC/AUC curves, and classification reports using Scikit-learn, Matplotlib, and Seaborn.

Model,Architecture Type,Objective
Model 1,Custom Baseline CNN,Establish baseline accuracy from scratch
Model 2,Transfer Learning,Harness pre-trained ImageNet representations
Model 3,Fine-Tuned Backbone,Optimize target class feature extraction

Tech Stack
Language: Python

Deep Learning: TensorFlow, Keras

Data & Image Processing: NumPy, Pandas, PIL (Pillow), imghdr

Visualization & Metrics: Matplotlib, Seaborn, Scikit-learn

1. Clone the Repository
git clone https://github.com/your-username/fast-food-image-classification.git
cd fast-food-image-classification
