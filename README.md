# Training TensorFlow (Keras) Deep Learning Models Efficiently

We present an **efficient methodology** to train Deep Learning models for image classification.

In practical Deep Learning projects, data is typically stored on disks. The **tf.Data** API provides faster solutions to load large data from the disk and apply various transformations on it. This API uses the **Dataset** object for representing a very large set of elements. It allows us to assert finer control over the data pipeline.

Using the tf.Data API, we construct a Dataset object from the local image repository. Then, the Dataset object is transformed for loading image-label pairs. Prior loading, we convert each encoded image (i.e., PNG-encoded images) as a Tensor object, type-casting it (e.g., float32), scaling, getting the image label from the stored images (from the image sub-directories organized by class names). Finally, image-label pairs are shuffled and put into batches for training the model. 

We use the **distributed training** technique. It utilizes multiple GPUs on a single node. The specification for the number and type of GPUs should be provided in the SLURM .sh job request file. More information on distributed training: https://keras.io/guides/distributed_training/



## Distributed Training: Single-host & Multi-device Synchronous Training

We use the **tf.distribute** API to train a TensorFlow Keras model on multiple GPUs installed on a single machine (node). 

Specifically, we use the tf.distribute.Strategy with tf.keras. The tf.distribute.Strategy is integrated into tf.keras, which is a high-level API to build and train models. By integrating into tf.keras backend, it is seamless for use to distribute the training written in the Keras training framework.

More information on the distributed training with Keras:
https://www.tensorflow.org/tutorials/distribute/keras


### Parallelism Technique:
Via the tf.distribute API, we implement the synchronous **data parallelism** technique. In this technique, a single model gets replicated on multiple devices or multiple machines. Each of them processes different batches of data, then they merge their results. The different replicas of the model stay in sync after each batch they process. Synchronicity keeps the model convergence behavior identical to what you would see for single-device training.


### How to use the tf.distribute API for distributed training:
To perform single-host, multi-device synchronous training with a TensorFlow Keras model, we use the tf.distribute.MirroredStrategy API. 

Following are the 3 simple steps.

- Instantiate a MirroredStrategy.
By default, the strategy will use all GPUs available on a single machine.

- Use the strategy object to open a scope, and within this scope, 
create all the Keras objects you need that contain variables. 
More specifically, within the distribution scope:
- Create the model
- Compile the model (by defining the optimizer and metrics)

- Train the model via the fit() method as usual (outside the scope).

**Note**: we need to use tf.data.Dataset objects to load data in a multi-device or distributed workflow.



## Local Data Repository

This program assumes that the training and test data (i.e., images) are stored locally, and organized in nested directories as follows:
- train
    - class_name_1
    - class_name_2
    
    ...
    
- test
     - class_name_1
     - class_name_2
     
   ...

Specifically, there should be two root directories named "train" and "test". The sub-directories should be named after the classes.

## Data Augmentation

In the distributed training setting, we recommend performing data augmentation inside the model so that it is done by GPUs. We use Keras image preproessing layer API to build the data augmentation pipeline. https://www.tensorflow.org/api_docs/python/tf/keras/layers/experimental/preprocessing

The augmentation layer can be used inside the data preprocessing pipeline when training is done by a single device (CPU or GPU).

Thus, while the data preprocessing pipeline is built using tf.Data API, the data augmentation pipeline is based on Keras image preproessing layer API, which can be integrated in the preprocessing pipeline when need be.

For a detail discussion on various augmentation approaches, see the following GitHub repository:
https://github.com/rhasanbd/Data-Augmentation-for-Deep-Learning


## Notebook Index

- Notebook 1: Data Loading & Preprocessing for Deep Learning by tf.Data API

- Notebook 2: An efficient methodology to train Deep Learning models for image classification.
