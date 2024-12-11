## Introduction

The task is to create and train a neural network that takes a random, variable-length array of real-valued numbers, and let it learn the L1-Norm of the array. Only the following operations are allowed: Dense Layers, Relu activations, the negation operation, and the sum and multiplication operations. Manually initializing a specific weight or set of weights are not allowed.

### Instructions
**install.sh** installs the dependencies and creates a local virtual environment to run the training.

**train.sh** starts training and reproduces the results.

### Method
I compared two different modeling approaches: (1) two dense layers with in-between ReLU non-linearity, namely L1NormLinear, (2) a recurrent neural network (RNN) either using a GRU cell (Gated Recurrent Unit), namely L1NormGRU, and a LSTM cell (Long Short-term Memory), namely L1NormLSTM. L1NormLinear approximates the abs operation per element, and then sums all outputs to predict the l1 norm. On the other hand, RNN models compute the abs function by ReLU(x) + ReLU(-x), and then updates the current hidden state and feeds it to the dense layer at each time step. The output of the dense layer accumulates the l1-norm prediction with the contribution from the current time step.

The maximum sequence length is an important factor for modeling. An RNN cell maintains a hidden state of a fixed size that should be determined before training. If we keep the maximum length very large but keep the number of RNN layers and hidden size small we might have underfitting issues. If we keep the maximum length small but keep the number of RNN layers and hidden size large we might have overfitting issues. Besides, growing the maximum sequence legth will result in more imbalance in the data due to having sequences of very different lengths. This will hamper the learning process, i.e., the model will see very long and very short sequences and it will have difficulty adapting its hidden state to these changes. We would probably see late convergence and zig-zaggings in the learning curve. The experiments tested different hidden sizes, datasets of different sizes with different maximum sequence length, and different input range. Mean squared error between the predicted and actual outputs were used to compare the performance of each configuration on the same test set.

A dataset of a given number of sequences, randomly selects a sequence length between 1 and the maximum sequence length, and then samples random floating point values in the uniform [min, max] range to fill the sequence. The sequences are then padded by zeros to maintain the same sequence length across each batch.

The data was organized as training, validation and test sets with 3:1 ratio between the training/validation and training/test. Each model was trained for a maximum of 50 epochs with 32 batch size. Early stopping according to the validation loss with 5 epoch patience was used. The optimizer was Adam with 1e-3 learning rate.

Implementation is done using PyTorch and Pytorch Lightning. Tensorboard is used to track the training and validation losses. I was able to use the Tensorboard directly in my Visual Studio Code enviroment. Installation in this repo is straightforward. Then, it should be run in the terminal and pointed to the log directory (./tb_logs).

### Results
The performance of the models were tested with 16, 32, and 64 hidden dimension size. The maximum sequence length was 20, 40, or 80, and the training sample size was either 30,000 or 80,000.

The test error of L1NormLinear was superior compared to all RNN models. The L1NormLinear training speed was also significantly faster than RNN models, since the model has simpler and less parameters. A detailed comparative analysis can be seen in the jupyter notebook, and custom comparisons also can be added there using the csv files in the output folder. 

With the 80 max sequence length and 64 hidden size the performance of the L1NormLSTM and the L1NormLinear was very close. L1NormLinear model training stopped earlier. This can be a hint that RNN models might be better for even higher sequence lengths given enough hidden size, because they are capable of utilizing long-term temporal relations better than a linear model having dense layers. To test this hypothesis I trained another L1NormLSTM and a L1NormLinear model with 500 max sequence length. Test error of L1NormLinear was 0.91 and LSTM was 1.50 (output/results_seqlen_500.csv). So, linear model was still performing better. To compensate the difficulty of higher sequence length, the hidden size and layers of the linear model can be increased accordingly. I trained another linear model with 500 max sequence length, this time with 128 hidden size and same layers. The test error was 0.35, which was better than the result with 64 hidden size (0.91).

### Conclusions and Next Steps
I trained a linear model and two rnn models to predict the l1 norm with only using the allowed operations. The linear model performed better and trained significantly faster than RNN models in my experiments. The next steps should explore sequence lengths greater than 500. To compensate the difficulties coming with the high sequence length model complexity, i.e., number of layers and hidden size, might need to be increased accordingly.
