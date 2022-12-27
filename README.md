# MD-RFA
A Multi-scale Dilation with Residual Fused Attention Network for Low-Dose CT Noise Artifact Reductions.

## Table of Contents
* [Introduction](https://github.com/kevinmfreire/MD-RFA#introduction)

## Introduction
Computed Tomography (CT) Scans have produced more than half the radiation exposure from medical use which results in problems for long term use of these expensive machines.  Some solutions have involved reducing the radiation dose, however that leads to noise artifacts making the low-dose CT (LDCT) images unreliable for diagnosis.  In this study, a Multi-scale Dilation with Residual Fused Attention (MD-RFA) deep neural network is proposed, more specifically a network with an integration with a multi-scale feature mapping, spatial- and channel-attention module to enhance the quality of LDCT images.  Further, the multi-scale image mapping uses a series of dilated convolution layers, which promotes the model to capture hierarchy features of different scales.  The attention modules are combined in a parallel connection and are described as a Boosting Attention Fusion Block (BAFB) that are then stacked on top of one another creating a residual connection known as a Boosting Module Group (BMG).

<figure>
    <img src="./images/MD-RFA.drawio.png" alt="MD-RFA Model Architecture">
    <figcaption>Figure 1: The Multi-scale Dilation with Residual Fused Attention (MD-RFA) Model Architecture.</figcaption>
</figure> 

<figure>
    <img src="./images/BoostingModules.drawio.png" alt="Boosting Module Groups">
    <figcaption>Figure 2: The Boosting Module Group composed of (A) Boosting Attention Fusion Block (BAFB), (B) Spatial Attention Module (SAM) and (C) Channel Attention Module (CAM).</figcaption>
</figure> 
