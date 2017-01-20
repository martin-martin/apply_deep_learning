[![Build Status](https://travis-ci.org/pkmital/CADL.svg?branch=master)](https://travis-ci.org/pkmital/CADL) [![Slack Channel](https://cadl.herokuapp.com/badge.svg)](https://cadl.herokuapp.com)

# <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>

This repository contains lecture transcripts and homework assignments as Jupyter Notebooks for the <a href="https://www.kadenze.com/partners/kadenze-academy">Kadenze Academy</a> course on <a href="https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info">Creative Applications of Deep Learning w/ Tensorflow</a>.

This course introduces you to deep learning: the state-of-the-art approach to building artificial intelligence algorithms. We cover the basic components of deep learning, what it means, how it works, and develop code necessary to build various algorithms such as deep convolutional networks, variational autoencoders, generative adversarial networks, and recurrent neural networks. A major focus of this course will be to not only understand how to build the necessary components of these algorithms, but also how to apply them for exploring creative applications. We'll see how to train a computer to recognize objects in an image and use this knowledge to drive new and interesting behaviors, from understanding the similarities and differences in large datasets and using them to self-organize, to understanding how to infinitely generate entirely new content or match the aesthetics or contents of another image. Deep learning offers enormous potential for creative applications and in this course we interrogate what's possible. Through practical applications and guided homework assignments, you'll be expected to create datasets, develop and train neural networks, explore your own media collections using existing state-of-the-art deep nets, synthesize new content from generative algorithms, and understand deep learning's potential for creating entirely new aesthetics and new ways of interacting with large amounts of data.

[Join our Slack channel.](https://cadl.herokuapp.com)

# Schedule

## Session 1: Introduction To Tensorflow
We'll cover the importance of data with machine and deep learning algorithms, the basics of creating a dataset, how to preprocess datasets, then jump into Tensorflow, a library for creating computational graphs built by Google Research. We'll learn the basic components of Tensorflow and see how to use it to filter images.

## Session 2: Training A Network W/ Tensorflow
We'll see how neural networks work, how they are "trained", and see the basic components of training a neural network. We'll then build our first neural network and use it for a fun application of teaching a neural network how to paint an image, and explore such a network can be extended to produce different aesthetics.

## Session 3: Unsupervised And Supervised Learning
We explore deep neural networks capable of encoding a large dataset, and see how we can use this encoding to explore "latent" dimensions of a dataset or for generating entirely new content. We'll see what this means, how "autoencoders" can be built, and learn a lot of state-of-the-art extensions that make them incredibly powerful. We'll also learn about another type of model that performs discriminative learning and see how this can be used to predict labels of an image.

## Session 4: Visualizing And Hallucinating Representations
This sessions works with state of the art networks and sees how to understand what "representations" they learn. We'll see how this process actually allows us to perform some really fun visualizations including "Deep Dream" which can produce infinite generative fractals, or "Style Net" which allows us to combine the content of one image and the style of another to produce widely different painterly aesthetics automatically.

## Session 5: Generative Models
The last session offers a teaser into some of the future directions of generative modeling, including some state of the art models such as the "generative adversarial network", and its implementation within a "variational autoencoder", which allows for some of the best encodings and generative modeling of datasets that currently exist. We also see how to begin to model time, and give neural networks memory by creating "recurrent neural networks" and see how to use such networks to create entirely generative text.

# Github Contents Overview

This github contains lecture transcripts from the Kadenze videos and homeworks contained in Jupyter Notebooks in the following folders:

| | Session | Description | Transcript | Homework |
| --- | --- | --- | --- | --- |
|Installation| **[Installation](#installation-preliminaries)** | Setting up Python/Notebook and necessary Libraries. | N/A | N/A |
|Preliminaries| **[Preliminaries with Python](session-0)** | Basics of working with Python and images. | N/A | N/A |
|1| **[Computing with Tensorflow](session-1)** | Working with a small dataset of images.  Dataset preprocessing.  Tensorflow basics.  Sorting/organizing a dataset. | [lecture-1.ipynb](session-1/lecture-1.ipynb) | [session-1.ipynb](session-1/session-1.ipynb) |
|2| **[Basics of Neural Networks](session-2)** | Learn how to create a Neural Network.  Learn to use a neural network to paint an image.  Apply creative thinking to the inputs, outputs, and definition of a network. | [lecture-2.ipynb](session-2/lecture-2.ipynb) | [session-2.ipynb](session-2/session-2.ipynb) |
|3| **[Unsupervised and Supervised Learning](session-3)** | Build an autoencoder.  Extend it with convolution, denoising, and variational layers.  Build a deep classification network.  Apply softmax and onehot encodings to classify audio using a Deep Convolutional Network. | [lecture-3.ipynb](session-3/lecture-3.ipynb) | [session-3.ipynb](session-3/session-3.ipynb) |
|4| **[Visualizing Representations](session-4)** | Visualize backpropped gradients, use them to create Deep Dream, extend Deep Dream w/ regularization.  Stylize images or synthesize new images with painterly or hallucinated aesthetics of another image. | [lecture-4.ipynb](session-4/lecture-4.ipynb) | [session-4.ipynb](session-4/session-4.ipynb) |
|5| **[Generative Models](session-5)** | Build a Generative Adversarial Network and extend it with a Variational Autoencoder.  Use the latent space of this network to perform latent arithmetic.  Build a character level Recurrent Neural Network using LSTMs.  Understand different ways of inferring with Recurrent Networks.  | [lecture-5.ipynb](session-5/lecture-5.ipynb) | [session-5-part-1.ipynb](session-5/session-5-part-1.ipynb), [session-5-part-2.ipynb](session-5/session-5-part-2.ipynb) |

<a name="installation-preliminaries"></a>

### Troubleshooting

Post on the [Forums](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow-i/forums?sort=recent_activity) or check on the Tensorflow [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/g3doc/get_started/os_setup.md#pip-installation)

## Notice
This code is part of the Kadenze Academy course on "Creative Applications of Deep Learning with Tensorflow" taught by Parag K. Mital.  More information can be found [on the course website](https://www.kadenze.com/courses/creative-applications-of-deep-learning-with-tensorflow/info) and the [course github](https://github.com/pkmital/CADL).

## Licensing Info
Distributed under the [Apache License](http://www.apache.org/licenses/) 2.0, January 2004