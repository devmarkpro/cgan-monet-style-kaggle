# CGAN Monet Style Transfer

This project implements a Conditional Generative Adversarial Network (CGAN) for generating Monet-style paintings from photographs, as part of the [Kaggle GAN Getting Started competition](https://www.kaggle.com/competitions/gan-getting-started/overview).

## Project Goal

The goal is to build a GAN that generates 7,000 to 10,000 Monet-style images. The model should transform regular photographs into paintings that mimic Claude Monet's impressionist style, characterized by:
- Loose brushwork and visible brushstrokes
- Emphasis on light and its changing qualities
- Movement and spontaneous composition
- Vibrant and pure colors
- Plein air (outdoor) painting techniques

## Dataset Specifications

The competition provides the following datasets:

### Training Data
- **Photo images**: 300 high-resolution photographs (JPEG format)
  - Landscape and outdoor scenes suitable for artistic transformation
  - Various lighting conditions and compositions
  
- **Monet paintings**: 300 high-resolution Monet paintings (JPEG format)
  - Authentic Claude Monet artworks
  - Diverse subjects including landscapes, water scenes, and garden scenes
  - Representative of different periods of Monet's artistic career

### Test Data
- **Photo images**: 7,038 photographs that need to be transformed into Monet-style paintings
- These images will be used to generate the final submission

### Technical Specifications
- **Image format**: JPEG
- **Typical resolution**: High-resolution images (exact dimensions vary)
- **Color space**: RGB
- **File structure**: Organized in separate directories for photos and monet paintings

The challenge is to learn the mapping between the photograph domain and the Monet painting domain using the provided training pairs, then apply this learned transformation to generate artistic renditions of the test photographs.

## Setup

This project is configured for `uv` package manager but also works with standard `pip`. To get started:

### Using uv (recommended):
```bash
uv pip install -e .
```

### Using pip:
```bash
pip install -e .
```

### Launch Jupyter notebook:
```bash
jupyter notebook
```

Then open `main.ipynb` to start experimenting with the model.

## Project Structure

```
cgan-monet-style-/
├── data/              # Dataset directory (add competition data here)
├── main.ipynb         # Main notebook for experimentation
├── pyproject.toml     # Project dependencies and configuration
└── README.md          # This file
```