# CNN Architecture Guide with PyTorch

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CNN-EE4C2C?logo=pytorch&logoColor=white)
![CNN](https://img.shields.io/badge/Deep%20Learning-CNN-8A2BE2)
![MNIST](https://img.shields.io/badge/Dataset-MNIST-4B8BBE)
![License](https://img.shields.io/badge/License-MIT-green)

<img align="center" width="100%" alt="cnn_guide_banner" src="https://github.com/user-attachments/assets/3ad785a4-ab08-4095-b8ca-cf04907c7f69" /> </br>

This repository provides a guide for building [*Convolutional Neural Networks (CNNs)*](https://en.wikipedia.org/wiki/Convolutional_neural_network) in PyTorch, aimed at beginners who want to understand how CNNs work and how to implement them. It combines theoretical explanations of key concepts from [Deep Learning](https://en.wikipedia.org/wiki/Deep_learning), such as the network architecture, Cross-Entropy Loss, and the Adam optimizer, with code that shows how these components come together in practice. <br/>
CNNs are widely used in [Computer Vision](https://en.wikipedia.org/wiki/Computer_vision) tasks, such as image classification, object detection, and image generation. 

This repository is built for learning purposes and helps beginners get started with coding neural networks and understanding their key components.

## Convolutional Neural Network (CNN)
A Convolutional Neural Network (CNN) is a type of feed-forward network that learns to perform tasks such as classification by optimizing its parameters (features). For each input, the network performs a forward pass, computing the outputs of each layer and storing intermediate activations. The loss is then calculated using the chosen loss function. During [*Backpropagation*](https://en.wikipedia.org/wiki/Backpropagation), the network updates its parameters using a gradient descent algorithm, moving toward a (local) minimum of the loss.

The network in this guide, referred to as the “Simple CNN”, consists of 6 layers: 2 convolutional layers, 2 pooling layers, and 2 fully-connected layers. It also employs dropout and batch normalization to improve generalization and training stability.

### Layers
In *Convolutional Neural Network* there are several types of layers, we will discuss the types that are relevant to our SimpleCNN model.
#### Fully-Connected Layer
<img align="right" height="300" alt="fc_cnn" src="https://github.com/user-attachments/assets/6c781d04-a4da-4fb7-b854-7b5183e22040" />
The Fully Connected (FC) layer consists of weights and biases, where every neuron in the output layer is connected to every neuron in the input layer through these weights.

(In the image, the input layer is shown in blue, the output layer in red, and the connecting arcs represent the weights.)

In this structure, each output neuron is influenced by all input neurons according to the corresponding weights. After this linear combination, a non-linear activation function is applied. The goal in this layer is to optimize both the weights and the biases. The formula below shows how to compute the $j-th$ output:

```math 
y_{j} = h \bigg( \sum_{i=1}^n w_{ji} x_i + w_{j0} \bigg) 
```
A network consisting solely of *Fully-Connected Layers* is called [*Multilayer perceptron*](https://en.wikipedia.org/wiki/Multilayer_perceptron).

#### Convolutional Layer
<img align="right" height="300" alt="conv_cnn" src="https://github.com/user-attachments/assets/916d2f0c-c225-44b1-88e3-292e53219f4f" />

The convolutional layer is a fundamental building block of a CNN. It performs a dot product between two matrices: one is the kernel (or filter), which contains learnable parameters, and the other is a portion of the layer’s input. The parameters to optimize in this layer are the kernels themselves.

For a more intuitive explanation of convolution, the [3Blue1Brown](https://www.youtube.com/watch?v=KuXjwB4LzSA) provides an excellent visual guide.

During a forward pass, each filter scans the input according to its specifications ([stride](https://deepai.org/machine-learning-glossary-and-terms/stride), [padding](https://deepai.org/machine-learning-glossary-and-terms/padding) etc.). At each position, a convolution (cross-correlation) operation is applied to the corresponding portion of the input to produce a single output value (as illustrated in the image). For multi-channel inputs, the operation is applied across all channels of the input, and the results are summed to produce the final output of the filter.

The size of the output can be calculated as follows:

```math
\\
H_{out} = \left\lfloor\frac{H_{in} - h_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1  \; ;\;
W_{out} = \left\lfloor\frac{W_{in} - w_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1
```


The number of output channels is the number of filters in the layer.

#### Pooling Layer
A pooling layer in Convolutional Neural Networks (CNNs) is used to reduce the spatial dimensions (height and width) of feature maps while retaining the most important information. The pooling operation applies a kernel that slides over the input, producing a single output value for each region, depending on the type of pooling (e.g., Max-Pooling or Average-Pooling, as illustrated in the figure below). The output size along each dimension is determined by the input size, kernel size, stride (step size), and padding (if applied).
```math
\\
H_{out} = \left\lfloor\frac{H_{in} - h_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1  \; ;\;
W_{out} = \left\lfloor\frac{W_{in} - w_{kernel} + 2 \cdot padding}{stride}\right\rfloor + 1
```
<br/>
<img align="center" width="2455" height="916" alt="pooling" src="https://github.com/user-attachments/assets/98c04a7f-16ab-4017-a5c7-76d2090ba686" />

### Activation Function
A neural network is a model used for function approximation and estimation. When non-linear activation functions are applied, even a simple two-layer network can approximate any continuous function, as stated by the [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem). To enhance the expressive power of neural networks, non-linear activation functions are applied after fully connected layers and convolutional layers, since convolution itself is a linear operation. These activations introduce non-linearity, enabling the network to capture complex patterns in the data.

Common examples of activation functions:
* Sigmoid - $`\sigma(x) = {1 \over {1+e^{-x}}}`$
* Hyperbolic Tangent - $`tanh(x) = {{e^x - e^{-x}} \over {e^x + e^{-x}}}`$
* Rectified Linear Unit (ReLU) - $`ReLU(x) = max(0,x)`$
* Leaky rectified linear unit (Leaky ReLU) - $`LReLU(x) = \begin{cases}0.01x \; & if & x \le 0 \\\ x \; & if & x > 0\end{cases}`$
* Softmax - $`Softmax(x)_{i} = \frac{e^{x_i}}{\sum_{j=0} e^{x_j}}`$


## Loss & Loss function
The loss measures the discrepancy between the predicted output of a model and the true target output according to a chosen criterion. It quantifies how well the model performs on a single example, reflecting the “error” or “cost” associated with that prediction. A loss function is a mathematical formulation that maps predictions and true targets to a single real number representing this cost. It serves as a guiding signal for learning, indicating how the model parameters should be updated to minimize the loss.

Loss functions are task-specific measures of the discrepancy between predicted and true outputs. For example, regression tasks often use Mean Squared Error (MSE) or Mean Absolute Error (MAE), while classification tasks commonly use Cross-Entropy or Hinge Loss. They can also encode penalties for particular mistakes, such as misclassifying certain classes or emphasizing outliers. Minimizing the loss over the dataset forms the optimization objective in machine learning and provides the signal used to compute gradients and update model parameters during training.

Common examples of loss functions:
* Mean Squared Error (MSE) - $`MSE = \frac{1}{N} \sum_{i=0} (y_i - t_i)^2`$
* Mean Absolute Error (L1 Loss) - $`MAE = \frac{\sum_{i=0} |y_i - t_i|}{N}`$
* Mean Bias Error - $`MBE = \frac{\sum_{i=0} (y_i - t_i)}{N}`$
* Hinge (SVM) - $`H_i = \sum_{j\neq y_i} max(0, s_j - s_{y_i}+1)`$
* Cross-Entropy - $`CE = -\frac{1}{N} \sum_{i=0} t_i \log{y_i}`$

Where $`y_i`$ is the model's prediction and $`t_i`$ is the true target (for Hinge, $`s_j`$ is the score of class $`j`$ and $`y_i`$ is the true class).

## Optimization
<img align="right" height="250" alt="optimization" src="https://github.com/user-attachments/assets/dcd5cca0-916a-4ac6-b26b-c488229cc06b" />

Mathematical optimization is the process of selecting the best element from a set of feasible options according to a given objective. An optimization problem is therefore the task of finding the solution that yields the minimum or maximum value of an objective function. Such problems can involve either continuous variables or discrete variables, depending on the context.

In machine learning, optimization refers to the process of adjusting a model’s parameters so that its predictions better match the target outputs. This is typically framed as minimizing a loss function, which quantifies the error between predicted and true values. Because modern models have millions of parameters and highly non-linear loss surfaces, analytical solutions are infeasible. Instead, iterative algorithms like [*Gradient Descent*](https://en.wikipedia.org/wiki/Gradient_descent) and its variants are used to gradually update parameters in the direction that reduces the loss.

Gradient descent is a first-order iterative optimization algorithm that uses the gradient of the loss function to guide parameter updates. Starting from an initial set of parameters, the algorithm computes the gradient of the loss with respect to each parameter and then moves the parameters in the opposite direction of the gradient, since this is the direction of steepest descent. The magnitude of each update is controlled by the learning rate (η), which determines how large a step is taken toward reducing the loss. Through repeated updates, the parameters gradually converge toward a local (or sometimes global) minimum of the loss function.

In the forward pass, the input propagates through the network to produce an output and compute the loss. In the backward pass (backpropagation), this loss is propagated backward through the network to determine how much each parameter contributed to the error. Using the chain rule, the gradients of the loss with respect to all weights and biases are calculated layer by layer. These gradients indicate how the parameters should change to reduce the loss. The Gradient Descent algorithm then updates the parameters in the opposite direction of the gradients. Repeating this cycle gradually improves the model by minimizing the loss function.

[3Blue1Brown](https://www.youtube.com/watch?v=Ilg3gGewQ5U) offers a clear and intuitive visualization of backpropagation.

The parameter update rule for the $i-th$ parameter at iteration $t + 1$ is:

```math
w^{(t+1)}_i = w^{(t)}_i - \eta·\nabla{L(w^{(t)}_i)}

```
Where 
* $`\eta`$ is the learning rate.
* $L$ is the Loss function.
* $`\nabla L`$ is gradient of the loss with respect to parameter $w_i$ 

The minimum conditions for applying gradient-based optimization are that the loss function must be [*Differentiable*](https://en.wikipedia.org/wiki/Differentiable_function) with respect to the model’s parameters and that gradients can be computed efficiently. Differentiability allows the use of the [*Chain rule*](https://en.wikipedia.org/wiki/Chain_rule), implemented through backpropagation, to propagate errors from the output layer back through the network. This ensures that each parameter receives an update signal that guides it toward reducing the loss.

Common variants of Gradient Descent:
- **Batch Gradient Descent** – uses the whole dataset per step.
- **Stochastic Gradient Descent (SGD)** – updates per single sample.
- **Mini-batch Gradient Descent** – compromise using small batches.
- **Momentum** – accelerates updates in consistent directions.
- **Adagrad** – adaptive learning rate based on past gradients.
- **RMSProp** – scales learning rates using moving averages.
- **Adam** – combines Momentum and RMSProp, widely used in deep learning.

### Regularization
Regularization refers to a set of techniques used to prevent a machine learning model from overfitting the training data, improving its generalization to unseen data. It works by constraining or penalizing the model’s complexity, encouraging simpler solutions that are less sensitive to noise in the data.

In our model, Simple CNN, we use *Dropout* and *Batch Normalization* methods.

#### Dropout
Dropout is a regularization technique where, during training, a fixed percentage of neurons (e.g. 50%) are randomly set to zero in each forward pass, preventing co-adaptation of neurons. This prevents over-reliance on specific neurons and encourages redundancy and robustness. <br/> To keep the expected activation unchanged, the remaining neurons are scaled by $`\frac{1}{1-p}`$ during training (*inverted dropout*, which is what PyTorch uses), so at inference time all neurons are active and no extra scaling is needed. <br/>

$$
\tilde{h_i} = \begin{cases}0 & \; & with & probability & p  \\\ \frac{h_i}{1-p} & \; & with & probability & 1-p\end{cases}
$$

During inference, all units are used as-is: $$\tilde{h_i} = h_i$$

#### Batch Normalization
Batch Normalization aims to stabilize and accelerate training by ensuring each channel’s activations have consistent statistics across mini‑batches. This method normalizes each feature channel’s activations to zero mean and unit variance over a mini-batch, thereby reducing internal covariate shift. It can also have a slight regularizing effect (due to batch noise). <br/> 
For a layer’s inputs $`x`$, we compute per‑channel mean, $`μ`$, and variance, $`σ^2`$, then transform: <br/>
```math
\hat{x} = \frac{(x - μ)}{\sqrt{σ^{2} + ε}}
```
Then we scale ($`γ`$) and shift ($`β`$):
```math
⇨  y = γ·\hat{x} + β
```
where $`γ`$ and $`β`$ are learned scale and shift parameters. This stabilizes and speeds up training and adds a bit of regularization through batch noise.


## The Dataset

<img align="right"  src="https://user-images.githubusercontent.com/34989887/204675687-03f39aeb-1039-4abc-aec5-7f1d6cbbe52e.png" width="350" height="350"/>

A dataset is a collection of data used to train, validate, and test machine learning models. It provides the examples from which the model learns patterns, relationships, or features relevant to a specific task, such as classification or regression. The quality, size, and diversity of the dataset directly influence the model’s performance and generalization ability, making dataset preparation and selection a critical step in any machine learning pipeline.

### *MNIST Dataset*
MNIST is a classic dataset of handwritten digits and is often referred to as the “hello world” of computer vision and machine learning. By providing a simple, standardized set of images paired with labels, it allows both newcomers and experts to quickly prototype and benchmark classification algorithms, explore feature learning, and compare new techniques against a familiar baseline.

This network is trained on the MNIST dataset, which consists of grayscale images of handwritten single-digit numbers (0–9). The task for the network is to classify each input image into the correct digit class. The dataset contains 70,000 images, with 60,000 images used for training and 10,000 images for testing. In this project, 20% of the training images (12,000) are held out as a validation set, leaving 48,000 for training.

For more information on the [MNIST Dataset](https://en.wikipedia.org/wiki/MNIST_database).

## The *Simple CNN* Model
Our model consists of 6 layers:
1. Convolution Layer with a kernel size of 5x5, and [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) activation function.
2. Max-pool Layer with a kernel size of 2x2.
3. Convolution Layer with a kernel size of 5x5, and ReLU activation function.
4. Max-pool Layer with a kernel size of 2x2.
5. Fully-connected Layer with input layer of 1024 and output layer of 512 and ReLU activation function.
6. Fully-connected Layer with input layer of 512 and output layer of 10 (classes). It outputs raw scores (logits); the [Softmax](https://en.wikipedia.org/wiki/Softmax_function) is applied inside `nn.CrossEntropyLoss`, so the model has no explicit Softmax layer.

<img width="3967" height="1296" alt="simpleCNNarchitecture" src="https://github.com/user-attachments/assets/a4b94761-60a1-4a42-a476-c9fd0642e526" />

The Simple CNN also uses methods to accelerate and stabilize the convergence of the network training, and avoid overfitting. 
After the second layer and fourth layer (Max-pool) the Simple CNN applies [*Dropout*](https://en.wikipedia.org/wiki/Dilution_(neural_networks)), and after the first layer and the third layer (Convolution) it applies [*Batch-Normalization*](https://en.wikipedia.org/wiki/Batch_normalization), before the activation.

### The Model with *pytorch*
The Simple CNN is implemented with [*pytorch*](https://pytorch.org/). In order to implement the network layers and methods pytorch module [*torch.nn*](https://pytorch.org/docs/stable/nn.html) is being used. Every layer/method apart from the fully connected layers gets a 4-dimensional input *(N,C,H,W)*, where *N* is the batch size, *C* is the number of the channels and *H,W* are height and width respectively, the resolution of the images.
There are multiple kinds of layers, methods and function that can be used from this module, and for the *Simple CNN* network we used:

* [**Conv2d**](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html) - Applies a 2D convolution over an input signal composed of several input planes. 
* [**MaxPool2d**](https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html) - Applies a 2D max pooling over an input signal composed of several input planes.
* [**Linear**](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html) - Applies a linear transformation to the layer's input, $`y = xA^T+b`$. In case of 4D input we flatten it to 2D, *(N,H)* / *(N,C·H·W)* with the same notations above.
* [**Dropout**](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html) - During training, randomly zeroes some of the elements of the input tensor with a given probability *p* using samples from a Bernoulli distribution. Each element is zeroed out independently on every forward call (unlike *Dropout2d*, which drops whole channels).
* [**BatchNorm2d**](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html) - Applies Batch Normalization over a 4D input, slicing through *C* (channel dimension), computing the mean ($`\mu`$) and variance ($`\sigma^2`$) over each *(N,H,W)* slice, and using these statistics to normalize that slice.

### Model Definition
```python

class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN) for MNIST classification.

    Architecture:
    - 2 Convolutional layers with ReLU and Batch Normalization.
    - 2 Max Pooling layers.
    - 2 Fully Connected (FC) layers.
    - Regularization:
        - 2 Dropout.
        - 2 Batch Normalization. 
    - Note: No explicit Softmax (applied by nn.CrossEntropyLoss).
    """

    def __init__(self, num_classes = 10):
        super(SimpleCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                               out_channels=conv1_out_channels,
                               kernel_size=conv_kernel_size)
        self.conv2 = nn.Conv2d(in_channels=conv1_out_channels,
                               out_channels=conv2_out_channels,
                               kernel_size=conv_kernel_size)

        # Max-Pooling layers
        self.max1 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
        self.max2 = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

        # Fully-Connected layers
        self.fc1 = nn.Linear(in_features=fc1_in, out_features=fc2_in)
        self.fc2 = nn.Linear(in_features=fc2_in, out_features=num_classes)

        # Dropout
        self.dropout1 = nn.Dropout(p=dropout1_rate)
        self.dropout2 = nn.Dropout(p=dropout2_rate)

        # Batch Normalization
        self.batch1 = nn.BatchNorm2d(num_features=conv1_out_channels)
        self.batch2 = nn.BatchNorm2d(num_features=conv2_out_channels)
```

### Training Epoch
```python
    for inputs, labels in train_loader:

        # Reset gradients
        optimizer.zero_grad()               

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, labels)   

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()
```

### Training Pipeline
```python
# Fix the random seed for reproducible results
set_seed(seed)

# Initialize model, loss function and optimizer
cnn_model, loss_fn, adam_optimizer, h_device = _setup_model_for_training(num_class, learning_rate)

# Initialize MNIST data loaders
train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size, validation_split)

# Train & Validation
train_losses, validation_losses = train_model(cnn_model, loss_fn, adam_optimizer, train_loader, val_loader, h_device)

# Test
test_accuracy, test_loss = evaluate_model(cnn_model, loss_fn, test_loader, h_device)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Plot Loss
plot_training_losses(train_losses, validation_losses)
```

### *Loss & Optimization*
#### Cross Entropy Loss Function
[**Cross Entropy Loss**](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) is widely used for classification tasks, as it measures the difference between the predicted probability distribution and the true distribution. Given a predicted probability vector **$\hat{y}$** and a one-hot encoded target vector **$y$**, the loss for a single example is defined as:

$$
\mathcal{L}_{CE} = - \sum_{i} y_i \log \hat{y}_i
$$

This loss penalizes confident incorrect predictions more heavily than less certain ones, encouraging the model to assign higher probabilities to the correct classes. Minimizing cross-entropy effectively maximizes the likelihood of the correct labels under the model’s predicted distribution.

#### Adam Optimizer
[**Adam (Adaptive Moment Estimation)**](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) is a widely used optimization algorithm in machine learning. It combines the benefits of Momentum and RMSProp, maintaining running estimates of both the mean and the uncentered variance of gradients to adaptively adjust the learning rate for each parameter.
By using these adaptive estimates, Adam can converge faster and more reliably on complex models, handle noisy gradients, and often requires less manual tuning of the learning rate compared to standard stochastic gradient descent.
Its adaptive nature makes Adam particularly effective for large-scale problems and deep neural networks, where gradients can vary significantly across parameters.

##### Adam Algorithm:
- $\theta_t$ : parameters at time step t.  
- $\beta_1, \beta_2$ : exponential decay rates for moment estimates.  
- $\alpha$ : learning rate.
- $\epsilon$ : small constant to prevent division by zero. <br/>


1. Compute gradients:
   <div align="center">
   $$g_t = \nabla_\theta J(\theta_t)$$
   </div>

2. Update moment estimates:
   <div align="center">
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad;\quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
   </div>
   
3. Bias correction: 
   <div align="center">
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \quad;\quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   </div>
   
4. Parameter update: 
   <div align="center">
   $$\theta_{t+1} = \theta_t - \alpha \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
   </div>

## Evaluation
The model performances are evaluated mainly by the *Loss*. Loss is the key measure of how far predictions deviate from the targets, driving the optimization process to adjust model parameters and minimize this error. Training loss reflects performance on the data used for learning, while validation loss measures performance on unseen data to assess generalization; a widening gap between them often indicates overfitting.

### Training & Validation Loss

<img width="2560" height="1335" alt="cnn4_g" src="https://github.com/user-attachments/assets/80137e24-bf7c-4785-b005-58216dd8c6db" />

### Typical Run

<img width="905" height="376" alt="cnn4_p" src="https://github.com/user-attachments/assets/4b4830a1-7e89-4705-9c33-2ac8d7b904b6" />

## Interactive Learning Path
**Open it at https://itumekanik.github.io/myz303/** (or download it from there as a ZIP and double-click `index.html`). The same site runs in two ways, from the same source in [web/](web/):

* **In the browser, with no server** (GitHub Pages, or the downloaded copy). Every computation runs in the browser: the CNN chapter's Inference and Theory tabs use the model trained with PyTorch here, and its Training tab trains SimpleCNN again on the reader's GPU with TensorFlow.js.
* **With PyTorch, on your computer**: [dashboard.py](dashboard.py) serves the same pages, and the CNN chapter then trains with `SimpleCNN.py`'s code in PyTorch (on an NVIDIA GPU through CUDA, otherwise on the CPU) instead of in the browser.

On Windows, double-click `start.bat`: the first time it sets up `.venv` with PyTorch (the CUDA build if there is an NVIDIA GPU) and builds the pages, then it opens http://127.0.0.1:8000/cnn.html. On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python dashboard.py
```

`dashboard.py` serves the pages built into `web/dist`. Build them once with Node.js 22.18 or newer (`cd web`, `npm install`, `npm run build`; `start.bat` does this when they are missing), or unpack the contents of the site's ZIP directly into `web/dist`, so that `web/dist/index.html` exists. After changing anything under `web/`, build again; [web/README.md](web/README.md) describes the development workflow and how the site is built and tested.

The CNN chapter (chapter 3):

* **Training tab** - a stage tracker (data loading → training → validation → test → saving), live batch loss, epoch loss and accuracy curves, the conv1 filters as they are updated, and predictions on 16 fixed validation images after every epoch. Every part has an (i) button that explains it.
* **Inference tab** - draw a digit with the mouse or pick an image from the test set, and see the class probabilities together with the feature maps of every layer (Conv1 → MaxPool → Conv2 → MaxPool → FC1 → FC2).
* **Theory tab** - the building blocks of the network (convolution, batch normalization, activation functions, pooling, dropout, softmax and cross-entropy, backpropagation and Adam), each explained next to a live demo that runs on a real MNIST image and the trained model's own weights: slide a kernel over the image step by step, compare the distributions before and after BatchNorm, drop units with a chosen rate, adjust the logits and watch the loss, or race SGD, Momentum and Adam on a loss surface.
* **Math tab** - the mathematics of every building block, from notation and SimpleCNN's shapes to convolution, batch normalization, activations, pooling, dropout, the fully connected layer, softmax with cross-entropy, backpropagation and the optimizers. Each section gives the forward formulas and derives the gradients step by step (revealed one line at a time), with interactive examples: the nine products behind a convolution output, BatchNorm on numbers you choose, gradient routing in max pooling, Dropout's expected value, a softmax gradient check against finite differences and Adam's first steps with and without bias correction.
* **Coding tab** - a step-by-step tutorial (in Turkish) that writes `SimpleCNN.py` from scratch: setup, imports, hyperparameters, data, the model, `forward`, the training and evaluation loops, saving the model and predicting, plus exercises. Code excerpts are taken from `SimpleCNN.py` when the pages are built, with editor-style syntax highlighting, real line numbers and numbered line notes.
* **Glossary** - the Glossary button in the top bar opens a searchable list of about 70 key terms (weight, bias, epoch, gradient, overfitting, receptive field, …), each with a short definition, what it looks like in SimpleCNN, and links to the chapter sections that show it. Terms underlined with dots in the explanations open the same definition in a small card.

### From the perceptron to the CNN
The site is a step-by-step learning platform. Click the page title (top left, with the ▾) to open the learning path: the chapters in historical order, with the current one marked. Every chapter has the same kind of tabs.

1. **Perceptron (1958)** - `perceptron.html`. Everything runs in the browser.
   * **Theory** - the artificial neuron as a clickable diagram that you turn into the AND, OR and NAND gates; the decision boundary as a line you can drag and turn; the perceptron learning rule applied one example at a time with every number shown; an experiment on how the margin sets the number of corrections; and why XOR cannot be learned, with the two-layer network that solves it.
   * **Playground** - place points (or pick AND, OR, XOR, clusters or a ring) and watch the perceptron learn, with the mistakes per epoch and the path of the weights.
   * **Math** - notation, the geometry of the boundary, why each correction helps, the perceptron rule as gradient descent, Novikoff's convergence theorem with its proof, and the proof that XOR is not linearly separable.
   * **Coding** - a walk through [Perceptron.py](Perceptron.py), a perceptron written from scratch with NumPy (`python Perceptron.py` trains it on AND, OR, NAND and XOR), plus the same in PyTorch and exercises.
2. **Multilayer network (MLP, 1986)** - `mlp.html`. Everything runs in the browser, on a small network written in JavaScript.
   * **Theory** - activation functions and their derivatives (why the step cannot learn); a 2-2-1 network learning XOR, shown both in the input space and in the hidden space where the points become separable; the loss as a function of one weight, with gradient descent stepping down it (cross-entropy against MSE, and a learning rate that is too large); backpropagation one step at a time on a 2-2-1 network, with every value and gradient; how many neurons it takes to fit a curve (universal approximation); and why an MLP is a poor fit for images, which leads to the CNN.
   * **Playground** - train a network with 0 to 3 hidden layers on XOR, a circle, two moons, a spiral or your own points; the diagram shows what every neuron has learned, and clicking a neuron shows it on the big plane.
   * **Math** - layers and the forward pass, the derivatives of the activations and the vanishing gradient, cross-entropy from maximum likelihood and its gradient, the four equations of backpropagation with a numerical gradient check, mini-batches and why the weights start random, and the universal approximation theorem.
   * **Coding** - a walk through [MLP.py](MLP.py), a network with one hidden layer written from scratch with NumPy (`python MLP.py` checks backpropagation against numerical derivatives, then learns XOR and the two moons), plus the same in PyTorch and exercises.
3. **Convolutional network (CNN)** - `cnn.html`, described above.
4. **Object detection and segmentation (YOLO)** - `yolo.html`. Two YOLO26n models, trained for this chapter on SpaceNet satellite tiles of Las Vegas, Paris, Shanghai and Khartoum, run in the browser with ONNX Runtime Web.
   * **Theory** - eight sections in two sessions, each with a live demo: the four tasks (classification, detection, semantic and instance segmentation) on one tile; boxes and IoU with a prediction box you drag; the three grids of 8400 candidates, cell by cell; the confidence threshold and NMS played step by step on the real model's candidates; masks from 32 prototypes, with a coefficient you change; training (label assignment, the three losses, pretraining, augmentation) with the same tile epoch by epoch; precision, recall and the PR curve on a Paris tile; and how much a one-pixel shift of the mask moves a building's area. Fourteen new glossary terms (IoU, NMS, mAP, …) link to these sections.
   * **Math** - notation and sizes (why 8400 = 21·640²/1024), how a box is decoded from a cell and four distances, IoU and the IoU losses (GIoU, CIoU) with their curves, sigmoid and binary cross-entropy against IoU-scaled targets, the mask loss and its gradient with respect to the coefficients (with gradient descent you can step), NMS and its cost, AP over 101 recall levels on a PR curve you edit, and how an outline's shift turns into an area error.
   * **Coding** - fifteen steps through the chapter's two files: [YOLO_SpaceNet_data.py](YOLO_SpaceNet_data.py) downloads the 746 SpaceNet tiles (about 1.9 GB; `--per-city 20` for a quick start), turns them into 8-bit JPEGs and builds both training sets with labels from OpenStreetMap; [YOLO_SpaceNet.py](YOLO_SpaceNet.py) trains (`train seg`, `train sem`, `--scratch`), reads the curves, measures, finds buildings and their areas on an image, and exports the ONNX file the page runs. Best run on Google Colab with a GPU; training takes about an hour.
   * **Practice** - 37 questions in the order of the Theory sections (IoU by hand, candidates for another input size, NMS on five boxes, precision and recall, areas in m², …) and ten on reading the training outputs (accuracy and its trap on imbalanced data, F1, our model's confusion matrix, overfitting in results.png), with feedback for typical wrong numbers, and five experiments on the trained model: a higher confidence threshold, a higher NMS threshold, the tile turned by 90°, without colour, and at a quarter of the resolution.
   * **Detection** - pick a validation tile or upload your own image: buildings and sports fields with their masks, the confidence and NMS thresholds as sliders (8400 candidates → those above the threshold → those left after NMS), building count, footprint area in m² and built-up ratio, a comparison with the true outlines (correct, wrong, missed), and a land-cover map (building, road, green, water, parking) with the share of each class.
   * **Training** - the record of our training runs: pretrained against from scratch, the three loss terms, precision, recall and mAP per epoch, the land-cover model's IoU per class, and the same tile predicted after 1, 5, 10, 25, 50, 75 and 100 epochs.
   * **Outputs** - how to read what Ultralytics writes during and after training, file by file, on our run's folder: when each file is written, the console lines column by column, results.csv and how best.pt is chosen, labels.jpg (class imbalance), the training and validation batch pictures, and the confusion matrix in four steps: TP, FP, FN and TN on a made-up post-earthquake damage survey with a threshold slider; a toy tile matched the way Ultralytics matches (why detection has no TN); our model's real matrix at any threshold from 0.001 (the one confusion_matrix.png is drawn at) to 0.95, with example crops for every cell; and the land-cover model's pixel matrix. Then the P, R, F1 and PR curves and how to choose a threshold. Eight new glossary terms (confusion matrix, TP, FP, FN, TN, F1, specificity, class imbalance).

The whole site can be switched between Turkish and English. Formulas are written in LaTeX and typeset with [KaTeX](https://katex.org) (MIT license), which is bundled into the pages, so the site also works without an internet connection.

With `dashboard.py`, the trained model is saved to `checkpoints/simple_cnn.pt` and loaded automatically the next time the dashboard starts. The run that produced it (loss curves, epoch table, filters, validation samples and log) is saved next to it in `checkpoints/simple_cnn_run.json`, so the Training tab shows the last run again after a restart. With the default settings, a dashboard run gives exactly the same result as `python SimpleCNN.py`.

## References
[The Back Propagation Method for CNN](https://ieeexplore.ieee.org/abstract/document/409626)

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

[Improving neural networks by preventing co-adaptation of feature detectors](https://arxiv.org/abs/1207.0580)

[Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)

