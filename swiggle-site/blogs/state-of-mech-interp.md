---
layout: page
title: The State of Mech Interp 2024
subtitle: September 10
permalink: /blogs/the-state-of-mech-interp-2024
use_math: true
---

### TLDR

We trained a VAE on the nouns dataset, 50k pixel art characters with glasses, unique heads, and t-shirts, and trained a sparse autoencoder to extract meaningful features represented in the latent space. We hand labelled 2048 features and created a playground for visualizing them and an interface for adjusting the strength of features in pixel art character. We created a RAG system for adding and removing features given a natural language prompt. This is a research prototype / exploration for future works in 1. interpreting diffusion models and 2. creating better interfaces for controlling their outputs beyond just text.

[Video Demo]

### Intro

Anthropic has done a lot of research in mechanistic interpretability for LLMs. They successfully trained a sparse autoencoder (SAE) on Claude Sonnet 3 and extracted millions of human interpretable features ranging from neuroscience, transit, to tourist attractions like the golden gate bridge. For Golden Gate Claude, they were able to get the LLM to behave like the Golden Gate Bridge by setting the activation of the “Golden Gate” feature to be high.

We applied their research to the latent space of VAEs as a precursor to interpreting diffusion models. Diffusion models are interesting because they’re another class of generative models and no one yet has attempted to peel back the black box around them. We believe this is important 1. it’s cool and 2. we think interfaces for prompting image models can be a lot better than just text and understanding how these models work could unlock new insights.

### Training the VAE

We trained the VAE on 50k pixel art characters from the nouns dataset. The VAE downscales the image through 4 conv layers with channel dimensions 64, 128, 512, and 1024. The model further compress the channel dimension to have a latent size of $4 \times S/16 \times S/16$. We train the model with a learning rate of $1e-3$ for 1,000,000 steps. We set the beta of the KL loss to be 0 for the first 1000 steps then linearly warm it up to 0.000001 to prevent posterior collapse. The input image size is 3 x 128 x 128 and the resulting latent size is 4 x 8 x 8.

### Training the SAE

We prepare the training data for the sparse autoencoder by computing the latents for each pixel art character in the dataset. We use the sparse autoencoder formulation from Anthropic.

The goal is to decompose the latent into a sparse linear combination of “features” which are vectors in the latent space that are semantically interesting. We train the model with an expansion factor of 8 (learning 2048 features) with no neuron resampling. We did a sweep across l1 values and carefully monitor reconstruction loss and the l0 norm (average number of features active for an image).

### Interpreting Features

To evaluate the features learned from the sparse autoencoder, we perform the following:

1. Create a one-hot vector $\mathbf{e}_i$ for feature $i$.
2. Pass $\mathbf{e}_i$ through the SAE decoder to get its representation in the VAE latent space $l_i = D_{SAE}(\mathbf{e}_i) = W_d \mathbf{e}_i + b_d$. This corresponds to the $i$th row of the decoder weight matrix plus the decoder bias.
3. Pass the latent vector $l_i$ through the VAE decoder to visualize the feature: $f_i = D_{VAE}(l_i)$.

Here is feature 983 visualized. We can reasonably deduce that it corresponds to the “crab head”.

<p style="text-align: center;">
  <img src="/assets/images/feature.png" alt="Human interpretable feature" />
</p>

To further corroborate that the feature matches our explanation, we collect the top k images with the highest feature activation and perform an ablations study to remove the feature vector from their latents.

<p style="text-align: center;">
  <img src="/assets/images/feature_topk_ablations.png" alt="Top K features ablations" />
</p>

We see that the top k images all contain the feature we’re hypothesizing and that the ablations confirm it. When we remove the “crab head” feature from the latent, it removes the “crab head” while preserving the shirt color and glasses color. (this is not true for this example, I’ll choose a better one)

### Features Playground

We created a playground to visualize the UMAP embeddings of the features. Check it out here: [link]

### Adjusting Feature Strength

We created a playground to dial up and down the features in a pixel art character. Experiment with what happens when you increase and decrease the strength of a feature. Check it out here: [link]

[image of feature playground]

### RAG on Latent Space

### What’s Next

Work on diffusion models.
