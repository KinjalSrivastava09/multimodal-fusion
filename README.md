# multimodal-fusion

Multimodal Fusion: Image–Text Representation Learning
A reproducible deep learning project exploring how visual and textual information can be combined to improve classification performance.
The project implements a Convolutional Neural Network (CNN) for image feature extraction, a lightweight text encoder, and a feature-level fusion network using PyTorch.
It uses a self-contained synthetic dataset, allowing the complete experiment to run without external datasets, pretrained models, or API keys.
Project Overview
Multimodal learning combines information from different sources to make predictions. This project investigates whether combining image and text representations can outperform models trained on either modality independently.
Three architectures are compared:
Image-only model: Uses a CNN to classify grayscale images.
Text-only model: Uses a feedforward encoder to classify controlled-vocabulary text features.
Multimodal fusion model: Combines image and text embeddings before classification.
An additional ablation experiment evaluates how the trained fusion model performs when its text embedding is removed during inference.
Experimental Results
All results were obtained by executing the included Python implementation.
Model
Test Accuracy
Image-only CNN
80.00%
Text-only encoder
77.11%
Multimodal fusion
100.00%
Fusion without text embedding
47.78%

Evaluation: 450 held-out synthetic test samples.
The fusion architecture achieves perfect classification on this particular synthetic test split. This demonstrates the benefit of complementary information in the controlled experimental setting.
The results should not be interpreted as evidence of equivalent performance on real-world multimodal datasets.
Dataset
The project generates 1,800 synthetic paired samples.
Property
Description
Total samples
1,800
Image resolution
24 × 24
Image channels
1 (grayscale)
Text representation
10-dimensional binary vector
Classification
Binary
Training samples
1,260
Validation samples
90
Testing samples
450

The classification task is intentionally constructed using complementary hidden information. The image and text modalities provide different pieces of evidence, allowing the fusion model to learn relationships that individual models cannot fully recover.
Model Architecture
Image Encoder
The image encoder uses a compact CNN consisting of convolutional layers, ReLU activations, max pooling, and a linear projection.
The resulting image embedding contains 16 features.
Text Encoder
The text encoder accepts a 10-dimensional binary feature vector and projects it into a 16-dimensional latent representation.
Multimodal Fusion
Both representations are concatenated:
[
h = [h_{\text{image}};h_{\text{text}}]
]
The resulting 32-dimensional representation is passed through a multilayer classifier to predict one of two classes.
Repository Structure
multimodal-fusion/
│
├── project.py
├── colab_demo.ipynb
├── README.md
├── requirements.txt
├── report.tex
├── report.pdf
│
├── figures/
│   └── Generated experimental figures
│
└── results/
    ├── metrics.json
    └── Trained model checkpoints
Run in Google Colab
The project can be executed in Google Colab without configuring a local Python environment.
Open colab_demo.ipynb from this repository in Google Colab.
Replace REPO_URL in the notebook's first code cell with your GitHub repository URL.
Select Runtime → Run all.
The notebook will install the required Python dependencies, execute the experiment, display the measured results, and show the generated figures.
No GPU, external dataset, pretrained weights, or LaTeX installation is required.
Run Locally
Clone the repository:
git clone https://github.com/USERNAME/REPOSITORY.git
cd REPOSITORY
Install dependencies:
python -m pip install -r requirements.txt
Run the experiment:
python project.py
Verify the saved outputs:
python project.py --check
The execution generates experimental outputs in the figures/ and results/ directories.
Visualizations
The project includes six generated figures covering the experimental setup and results, including:
Synthetic image examples
Multimodal fusion architecture
Model training dynamics
Classification accuracy comparison
Confusion matrix
Latent representation visualization using PCA
These figures are generated from the implemented experiment rather than manually constructed performance examples.
Reproducibility
The experiment uses:
PyTorch
Fixed random seed (20261030)
Adam optimizer
Learning rate: 0.003
Training epochs: 12
Batch size: 64
Cross-entropy loss
All three models share the same training, validation, and testing partitions.
Predictions, confusion matrices, accuracy scores, training histories, and model checkpoints are saved for inspection.
The verification command checks the consistency of stored predictions and reported evaluation results.
Technical Report
A detailed research report is included in two formats:
report.pdf — formatted technical report
report.tex — editable LaTeX source
The report documents the methodology, neural network architecture, mathematical formulation, experiments, results, limitations, and conclusions.
The LaTeX source is maintained separately from the Python implementation. Running the experiment does not require LaTeX or automatically update the report.
Limitations
This project is intended as an educational and controlled research demonstration.
The dataset is synthetic, and its complementary information structure is deliberately designed. The text encoder does not process unrestricted natural language, and the experiment has not been validated on real-world multimodal datasets.
Future research could investigate cross-modal attention, modality dropout, noisy or missing inputs, real image–text datasets, and repeated experiments across multiple random seeds.
Conclusion
This project demonstrates how combining two imperfect information sources can improve classification performance when their features contain complementary evidence.
The experiment provides a compact, reproducible implementation of multimodal representation learning, together with evaluation metrics, visualizations, trained models, and a technical report.

Framework: PyTorch
Language: Python
Research area: Deep Learning, Multimodal Learning, Representation Learning
Dataset: Synthetic paired image–text data
Project type: Reproducible experimental study

