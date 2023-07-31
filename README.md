<!-- PROJECT LOGO -->

<p align="center">

  <h1 align="center">Leveraging BERT for Enhanced Tweet Sentiment Analysis</h1>
  <p align="center">
    <a href="https://ch.linkedin.com/in/khanhvu207"><strong>Khanh Vu</strong></a>
    ·
    <a href="https://ch.linkedin.com/in/changlingli1998"><strong>Changling Li</strong></a>
    ·
    <a href="https://zzh2000.github.io"><strong>Zihan Zhu</strong></a>
    ·
    <a href="https://www.xccyn.com/"><strong>Xin Chen</strong></a>
  </p>
  <p align="center"><strong>Group: Attention is all you need</strong></p>
  <p align="center"><strong>Department of Computer Science, ETH Zurich, Switzerland</strong></p>
  <!-- <h2 align="center"></h2> -->
  <div align="center"></div>
</p>
<br>



<br>

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#introduction">Introduction</a>
    </li>
    <li>
      <a href="#data">Data</a>
    </li>
    <li>
      <a href="#machine-learning-baselines">Machine Learning Baselines</a>
    </li>
    <li>
      <a href="#deep-learning-baselines">Deep Learning Baselines</a>
    </li>
    <li>
      <a href="#bert-finetuning">BERT Finetuning</a>
    </li>
    <li>
      <a href="#contact">Contact</a>
    </li>
  </ol>
</details>


## Introduction
Large language models have revolutionized various fields, showcasing their capacity for understanding human natural languages. Tweet sentiment analysis is a challenging and valuable task within its context, given its direct relevance to social media analyses. In this report, we delve into a method utilizing BERT (Bidirectional Encoder Representations from Transformers) ensemble, showcasing its efficiency in achieving commendable performance in tweet sentiment analysis compared to several standard baselines. We conduct a comprehensive suite of experiments, with discussions on the critical role of data preprocessing in improving model performance. Our findings provide insights into the development of more robust and efficient sentiment analysis models.

We include baseline and our methods' implementation in the following sections.

## Data
The preprocessed data can be downloaded [here](https://drive.google.com/file/d/1YNJAKRipuUkPN9yxvgkK9CMYcg6Ou_kQ/view?usp=sharing), remember to decompress and put it under `src/data`.

## Machine Learning Baselines
Please check [src/ML/](src/ML/).
## Deep Learning Baselines
Please check [src/DL/](src/DL/).
## BERT Finetuning
Please check [src/bert-fine_tuning/](src/bert-fine_tuning/).

## Contact
Contact [Khanh Vu](mailto:khanvu@ethz.ch), [Changling Li](mailto:lichan@ethz.ch), [Zihan Zhu](mailto:zihzhu@ethz.ch) and [Xin Chen](mailto:chexin@ethz.ch) for questions, comments and reporting bugs.