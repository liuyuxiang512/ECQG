# Entity-Centric Question Generation

This repository contains the code and dataset for the *Findings of EMNLP 2023* paper: "Ask To The Point: Open-Domain Entity-Centric Question Generation". [[link](https://arxiv.org/abs/2310.14126)]

ECQG dataset is available in [Google Drive](https://drive.google.com/drive/folders/1akNKLzoTu61UUv0IMd7D6hv6pldgOfMt?usp=share_link). 

## Introduction

We introduce a new task called *entity-centric question generation* (ECQG), motivated by real-world applications such as topic-specific learning, assisted reading, and fact-checking. The task aims to generate questions from an entity perspective. To solve ECQG, we propose a coherent PLM-based framework GenCONE with two novel modules: content focusing and question verification. The content focusing module first identifies a focus as "what to ask" to form draft questions, and the question verification module refines the questions afterwards by verifying the answerability. We also construct a large-scale open-domain dataset from SQuAD to support this task. Our extensive experiments demonstrate that GenCONE significantly and consistently outperforms various baselines, and two modules are effective and complementary in generating high-quality questions. 

![Model Overview](https://github.com/liuyuxiang512/ECQG/blob/main/GenCONE_framework.png)

## Update

[2023/10/21] Upload code and dataset. 

## Cite

> TBD
