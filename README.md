# Deep Learning Baselines

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#data">Data</a>
    </li>
    <li>
      <a href="#glove-embedding">GloVe Embedding</a>
    </li>
    <li>
      <a href="#run">Run</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>

## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `DNN`.
```bash
conda env create -f environment.yaml
conda activate DNN
```

## Data
The preprocessed data can be downloaded [here](https://drive.google.com/file/d/1YNJAKRipuUkPN9yxvgkK9CMYcg6Ou_kQ/view?usp=sharing), remember to decompress and put it under `src/data`.

## GloVe Embedding
The GloVe Embedding can be downloaded [here](https://www.kaggle.com/datasets/bertcarremans/glovetwitter27b100dtxt/download?datasetVersionNumber=1), remember to put it under `src/ML/embedding`.

## Example training code

python3 DNN_classifier.py

