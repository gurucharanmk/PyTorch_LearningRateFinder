# PyTorch_LearningRateFinder

PyTorch implementation of learning rate finder to find an optimal learning rate for a PyTorch model,as described in Leslie Smith's paper: https://arxiv.org/abs/1506.01186

Learning rate as being one of the key hyper-parameter in neural network, finding optimal learning rate for a deep network is very tricky. A low learning rate may take very long to converge against an optimal solution, while a higher learning rate quickly diverges, but may never find the best solution.

Leslie Smith's paper: https://arxiv.org/abs/1506.01186 describes one of the best solution to get optimal learning rate.
