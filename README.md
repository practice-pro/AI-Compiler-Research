# AI-Compiler-Research

This repository documents my ongoing research on **AI compiler optimization** that originated from my participation in the **Kaggle contest "Google - Fast or Slow? Predict AI Model Runtime"**. I started this work as a 4th-year undergraduate student in Computer Science in 2023, where I applied machine learning methods to predict AI model runtimes. The project continues to explore AI-driven approaches for optimizing compilers and predicting the runtime of AI models across different hardware configurations, especially focusing on **TPUs**.

## Project Overview

In this project, I implemented a **Graph Convolution Network (GCN)** during the Kaggle contest to predict the runtime of various AI models based on their computational graph representation. The contest provided an opportunity to apply **AI-driven techniques** for **compiler optimizations** and performance prediction, and this repository serves as an extension of that work.

The ongoing research aims to expand on the Kaggle model by exploring additional machine learning techniques, refining the existing approach, and integrating more complex datasets.

## Discoveries Through the Contest

I am deeply grateful to the organizers of the Kaggle contest for this invaluable learning experience. The contest allowed me to explore how machine learning models could be used for **compiler optimizations**, focusing on predicting the execution runtime of AI models. During the contest, I successfully built and tested a **Graph Convolution Network (GCN)** model that predicted runtime classes for different hardware configurations.

## Background

The Kaggle contest was inspired by the **paper "TpuGraphs: A Performance Prediction Dataset on Large Tensor Computational Graphs"**, which describes the computational graph representation of programs running on **TPUs**. This work provided insights into how the compilation configuration can affect the runtime, and I built on this by using machine learning techniques to predict the runtime performance of AI models based on their computational graphs.

## Approach

### Computational Graph Representation

In this work, AI models are represented as **computational graphs**, where nodes represent the operations in the model, and edges capture the dependencies between these operations. The goal was to predict the order of execution that leads to optimal performance based on these graph representations.

### Graph Convolution Network (GCN)

The **Graph Convolution Network (GCN)** is used to predict the order of execution for each node based on its features. The network was trained to predict runtime classes (fastest to slowest) by analyzing the configuration of nodes and edges in the graph.

### Key Concepts:
- **Node Features**: Each node in the computational graph is assigned a feature vector that represents its computational characteristics.
- **Opcode for Tiles**: Refers to the operations encoded for various hardware tiles in the computational graph.
- **Normalization**: The feature vectors are normalized based on their influence on the overall runtime.

### Training and Prediction:
The **GCN model** was trained to predict which configurations would result in faster or slower execution times based on the graph structure. It produced an **n x n matrix** representing the likelihood of each configuration belonging to a specific runtime class, helping identify the optimal order of execution.

## Data Preparation

### Feature Weight Assignment:
Feature vectors are weighted in the range of **[0.0055-0.01]** based on the specific instructions represented in the computational graph.

### Runtime Calculation:
The **runtime for each node** is calculated as the weighted sum of its feature vectors, providing an estimate of the overall execution time. These calculated runtimes were then normalized to help the model classify nodes from the fastest to the slowest runtime.

### GCN Training:
The GCN was trained on feature vectors derived from the computational graph and used **edge vectors** to learn the dependencies between the nodes, resulting in an accurate prediction of execution order.

## Model Interpretation

The trained GCN outputs an \( n \times n \) matrix, where \( n \) is the total number of configurations. Each entry \( \text{pred}[i][j] \) represents the probability of configuration \( i \) belonging to runtime class \( j \):

- **Class 0**: Fastest runtime.
- **Class \( j \)**: The \( j \)-th fastest runtime.
- **Class \( n-1 \)**: Slowest runtime.

For a configuration \( i \):
- The matrix entry with the highest probability in row \( i \) indicates the predicted runtime class.
- Example: If pred\[i\]\[j\] is the highest value, configuration \( i \) belongs to class \( j \), i.e., it is predicted to have the \( j \)-th fastest runtime.
  
The approach bears similarities to the "bag-of-words" model used in text classification, where feature vectors are used to classify Machine Learning keywords in datasets such as the Cora dataset. Here, feature vectors represent nodes and edges in computational graphs, which are used to determine runtime classes instead of text categories.

## Ongoing Work

While the GCN model has already been trained and tested during the Kaggle contest, this repository continues to evolve as I build upon this work. The next steps involve:
- Expanding the dataset to include more complex configurations.
- Investigating additional machine learning techniques to improve accuracy and performance predictions.
- Implementing a GRU-based Runtime Prediction Model: As part of improving the prediction of model runtimes, I have developed a GRU-based model to better capture the sequential dependencies in the data and enhance runtime prediction accuracy. This model includes multiple layers, such as an input layer for node features and a GRU layer for capturing complex temporal patterns in runtime data. The goal is to make runtime predictions more robust, especially when dealing with variable node features and different operation codes (opcodes).
- Integrating the model into an AI compiler framework for real-time optimization of runtime.

## Acknowledgments

This repository utilizes the following dataset:
- **Google - Fast or Slow? Predict AI Model Runtime:**:  
   Kaggle competition dataset (2023) by Mangpo Phothilimthana, Sami Abu-El-Haija, Bryan Perozzi, Walter Reade, and Ashley Chow.  
   [View the Dataset on Kaggle](https://www.kaggle.com/competitions/predict-ai-model-runtime).  

For citation, use the following BibTeX entry:
```bibtex

 @misc{predict-ai-model-runtime,
    author = {Mangpo Phothilimthana and Sami Abu-El-Haija and Bryan Perozzi and Walter Reade and Ashley Chow},
    title = {Google - Fast or Slow? Predict AI Model Runtime},
    year = {2023},
    howpublished = {\url{https://kaggle.com/competitions/predict-ai-model-runtime}},
    note = {Kaggle}
 }
