# Hybrid Models for Facial Emotion Recognition in Children

This repository contains code and resources for the paper titled "Hybrid Models for Facial Emotion Recognition
in Children." The paper focuses on the use of emotion recognition techniques to assist
psychologists in performing children's therapy through remotely robot operated sessions.

## Abstract

In the field of psychology, the use of agent-mediated therapy is growing increasingly given recent advances in robotics
and computer science. Specifically, the use of Embodied Conversational Agents (ECA) as an intermediary tool can help
professionals connect with children who face social challenges such as Attention Deficit Hyperactivity Disorder (ADHD),
Autism Spectrum Disorder (ASD), or who are physically unavailable due to being in regions of armed conflict, natural
disasters, or other circumstances. Emotion recognition plays a crucial role in providing valuable feedback to the
psychotherapist.

This paper presents the results of a bibliographical research associated with emotion recognition in children. The
research provides an overview of algorithms and datasets widely used by the community. Based on the analysis of the
bibliographical research results, we propose a technique using dense optical flow features to improve the ability to
identify emotions in children in uncontrolled environments. The proposed architecture, called HybridCNNFusion, utilizes
a hybrid model of Convolutional Neural Network (CNN) where two intermediary features are fused before being processed by
a final classifier. The initial results achieved in the recognition of children's emotions are presented using a dataset
of Brazilian children.

## Folder Structure

The repository has the following folder structure:

* __'checkpoints/'__: This directory is for saved pre-trained models and checkpoints for the emotion recognition system.
* __'data/'__: This directory is for storing the datasets used in the experiments, including the dataset of Brazilian
  children.
* __'src/'__: This directory contains the source code for the emotion recognition system, including the implementation
  of the HybridCNNFusion architecture.
* __'requirements.txt'__: This file lists the dependencies and libraries required to run the code in this repository.

## Usage

To use the code in this repository, follow these steps:

Clone the repository:

1. `git clone https://github.com/your-username/repo-name.git`

2. `pip install -r requirements.txt`

3. `python -m src.train` or `python -m src.predict`

This will run the system and provide the recognition results based on the provided dataset.

## License

The code and resources in this repository are available under the MIT License. Feel free to modify and adapt them for
your own projects.

## Considerations

The above information is a summary of the paper. For a complete understanding, please refer to the full paper.