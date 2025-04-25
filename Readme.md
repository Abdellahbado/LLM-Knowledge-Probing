# Knowledge Probing for Continent Classification

This project probes the `google/gemma-3-4b-it` language model to assess its ability to associate countries with their continents using logistic regression classifiers trained on hidden state representations from various layers.

## Overview

- **Model**: `google/gemma-3-4b-it`
- **Task**: Predict a country's continent based on its name.
- **Method**: Extract hidden state representations at specified layers, train logistic regression probes, and evaluate performance.
- **Layers Probed**: 0, 4, 8, 12, 16, 20, 24, 28, 31
- **Data**: 50 training countries (10 per continent: Europe, Asia, Africa, North America, South America) and 10 test countries (2 per continent).

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Scikit-learn
- NumPy
- Matplotlib
- Seaborn
- Plotly
- Pandas

Install with:
```bash
pip install -r requirements.txt
```

## Usage

1. Set your Hugging Face token (if required):
   ```bash
   export HF_TOKEN=your_token_here
   ```
2. Run the script:
   ```bash
   python probe.py
   ```
3. Outputs are saved to the `plots` directory, including accuracy plots, dendrogram, heatmap, PCA visualization, and error analysis.

## Outputs

- **Layer-wise Accuracy Plot**: Test accuracy and CV scores across layers.
- **Similarity Heatmap**: Cosine similarity of representations at the best layer.
- **Dendrogram**: Hierarchical clustering of country representations at the best layer.
- **3D PCA Plot**: 3D visualization of representations.
- **Continent Accuracy Bar Chart**: Accuracy per continent.
- **Error Analysis**: Misclassification details.

## Notes on Layer-wise Accuracy and Dendrogram

- **Layer-wise Accuracy**:
  - Accuracy peaks at layers 8, 27, and 30 (1.00), with a notable dip at layer 20 (0.60), indicating varying semantic capture across layers. 

- **Dendrogram (Layer 8)**:
  - Visualizes clustering of country representations with distances from 0.00 to 0.04.

  - Clusters generally align with continents(Algeria, Tunisia, Morocco), though some cross-continental similarities may appear.