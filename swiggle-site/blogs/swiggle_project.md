---
layout: page
title: Swiggle
subtitle: August - September 2024
permalink: /swiggle/blogs/swiggle_project
use_math: true
---

### TLDR

We trained a VAE on the nouns dataset, 50k pixel art characters with glasses, unique heads, and t-shirts, and trained a sparse autoencoder to extract meaningful features represented in the latent space. We hand labelled 2048 features and created a playground for visualizing them and an interface for adjusting the strength of features in pixel art character. We created a RAG system for adding and removing features given a natural language prompt. This is a research prototype / exploration for future works in 1. interpreting diffusion models and 2. creating better interfaces for controlling their outputs beyond just text.

<iframe src="https://drive.google.com/file/d/1PA2WM6LP9WeZK2NsODQlzTsr1991Tb6T/preview" width="640" height="360" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### Intro

Anthropic has done a lot of research in mechanistic interpretability for LLMs. They successfully trained a sparse autoencoder (SAE) on Claude Sonnet 3 and extracted millions of human interpretable features ranging from neuroscience, transit, to tourist attractions like the golden gate bridge. For Golden Gate Claude, they were able to get the LLM to respond as if it were the Golden Gate Bridge by increasing the activation strength of the “Golden Gate” feature.

We applied their research to the latent space of VAEs as a precursor to interpreting diffusion models. Diffusion models are interesting because they’re another class of generative models and no one yet has attempted to peel back the black box around them. We believe this is important 1. it’s cool and 2. we think interfaces for prompting image models can be a lot better than just text and understanding how these models work could unlock new insights.

### Training the VAE

We trained the VAE on 50k pixel art characters from the [nouns dataset](https://huggingface.co/datasets/m1guelpf/nouns). The VAE downscales the image through 4 conv layers with channel dimensions 64, 128, 512, and 1024. The model further compress the channel dimension to have a latent size of $4 \times S/16 \times S/16$. We train the model with a learning rate of $1e-3$ for 2,000,000 steps. We set the beta of the KL loss to be 0 for the first 1000 steps then linearly warm it up to 0.000001 to prevent posterior collapse. The input image size is 3 x 128 x 128 and the resulting latent size is 4 x 8 x 8.

<figure style="text-align: center; width: 100%;">
  <img src="/assets/images/original_vs_reconstructed.jpeg" alt="Original vs. Reconstructed Pixel Art Characters at Iteration 2,000,000" style="max-width: 100%;" />
  <figcaption style="color: gray; font-style: italic; display: block; width: 100%;">Original vs. Reconstructed Pixel Art Characters at Iteration 2,000,000</figcaption>
</figure>

### Training the SAE

We prepare the training data for the sparse autoencoder by computing the latents for each pixel art character in the dataset. We use the sparse autoencoder formulation from Anthropic.

The goal is to decompose the latent into a sparse linear combination of “features” which are vectors in the latent space that are semantically interesting. We train the model with an expansion factor of 8 (learning 2048 features) with no neuron resampling. We did a sweep across l1 values and carefully monitor reconstruction loss and the l0 norm (average number of features active for an image).

<div style="display: flex; justify-content: center; align-items: flex-start; gap: 20px; width: 55%">
  
  <figure style="text-align: center; width: 50%;">
    <img src="/assets/images/l0_norm.jpeg" alt="L0 Norm measures the average number of active neurons across all pixel art characters" style="max-width: 100%;" />
    <figcaption style="color: gray; font-style: italic; display: block; max-width: 100%;">L0 Norm measures the average number of active neurons across all pixel art characters</figcaption>
  </figure>

  <figure style="text-align: center; width: 50%;">
    <img src="/assets/images/feature_density.jpeg" alt="Histogram of log feature density" style="max-width: 100%;" />
    <figcaption style="color: gray; font-style: italic; display: block; max-width: 100%; text-align: center;">Histogram of log feature density</figcaption>
  </figure>

</div>

### Interpreting Features

To evaluate the features learned from the sparse autoencoder, we perform the following:

1. Create a one-hot vector $\mathbf{e}_i$ for feature $i$.
2. Pass $\mathbf{e}_i$ through the SAE decoder to get its representation in the VAE latent space $l_i = D_{SAE}(\mathbf{e}_i) = W_d \mathbf{e}_i + b_d$. This corresponds to the $i$th row of the decoder weight matrix plus the decoder bias.
3. Pass the latent vector $l_i$ through the VAE decoder to visualize the feature: $f_i = D_{VAE}(l_i)$.

Here is feature 176 visualized. We can reasonably deduce that it corresponds to the “factory head”.

<p style="text-align: center;">
  <img src="/assets/images/feature.png" alt="Human interpretable feature" />
</p>

To further corroborate that the feature matches our explanation, we collect the top k images with the highest feature activation and perform an ablations study to remove the feature vector from their latents.

<p style="text-align: center;">
  <img src="/assets/images/feature_topk_ablations.png" alt="Top K features ablations" />
</p>

We see that the top k images all contain the feature we’re hypothesizing and that the ablations confirm it. When we remove the “factory head” feature from the latent, it removes the “factory head” while preserving the shirt color and glasses color.

### Features Playground

We created a playground to visualize the UMAP embeddings of the features. Here’s a video of how it works. One neat thing is how the embeddings of the features separated itself into 3 clusters (heads, shirts, and glasses).

<iframe src="https://drive.google.com/file/d/19T5AqDrf3sR4XqXiBFUH9rKnAXNrF67k/preview" width="640" height="360" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### Adjusting Feature Strength

We created a playground to dial up and down the features in a pixel art character. Experiment with increasing and decreasing the strength of a feature.

<iframe src="https://drive.google.com/file/d/16Ett_JnyOh-aMA0W_BHymw_LsYGlToC2/preview" width="640" height="360" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### RAG on Latent Space

We generated text embeddings for the hand labeled features and implemented a RAG system to edit pixel art characters using natural language prompts. Given a prompt, we semantically search for the relevant features and decide whether to increase or decrease its activation. This all works without training a new model to follow instructions!

<iframe src="https://drive.google.com/file/d/1qlb5QKXVsCjPX1XvdqeFsJw5nISj_-Hc/preview" width="640" height="360" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### What’s Next?

Work on diffusion models.
