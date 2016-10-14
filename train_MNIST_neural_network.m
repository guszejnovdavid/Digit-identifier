function [NeuralNetwork] = train_MNIST_neural_network()

tic;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MNIST 60K database for training
trainingData.images = load_MNIST_images('data/train-images.idx3-ubyte');
trainingData.labels = load_MNIST_labels('data/train-labels.idx1-ubyte');
trainingData.numImages = length(trainingData.images);

% MNIST 10K database for testing
testData.images = load_MNIST_images('data/t10k-images.idx3-ubyte');
testData.labels = load_MNIST_labels('data/t10k-labels.idx1-ubyte');
testData.numImages = length(testData.images);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Initialization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Activation Functions
    
% Functions for neural networks
SigmoidFunction = @(z) 1./(1 + exp(-z)); %sigmoid
%TanhFunction   = @(z) tanh(z); %tanh

% Activation function of neuron
NeuralNetwork.ActivationFunction = @(z) SigmoidFunction(z);
% Derivative of activation function
NeuralNetwork.ActivationFunctionDeriv = @(z) SigmoidFunction(z).*(1-SigmoidFunction(z));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Parameters
    
% Neural network initialization
% Number of layers and number of neurons in each layer
NNodes=30; %number of nodes in the hidden layer
NeuralNetwork.NNeuron = [size(trainingData.images,1),NNodes,length(unique(trainingData.labels))];
NeuralNetwork.NLayer = length(NeuralNetwork.NNeuron);
% Initialize network biases and weights
for i = 2:NeuralNetwork.NLayer
  NeuralNetwork.bias{i}=randn(NeuralNetwork.NNeuron(i),1);
  NeuralNetwork.weight{i}=randn(NeuralNetwork.NNeuron(i),NeuralNetwork.NNeuron(i-1))./sqrt(NeuralNetwork.NNeuron(i));
end

% Regularization of the cost function
NeuralNetwork.RegularizationLambda = 1.0e-8;

  % Gradient descent
GradientDescent.stepSize = 1.00; %step size
% Number of images used in each epoch to estimate gradients
GradientDescent.NTrainImages = 30;
% Number of gradient descent epochs
GradientDescent.NEpochs = 1000;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Train network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i = 1:GradientDescent.NEpochs %for each epoch
  % Randomly pick training images
  ChosenTrainingImages =randi([1 length(trainingData.images)],1,GradientDescent.NTrainImages);
  %Estimate gradient
  for j = 1:GradientDescent.NTrainImages
    % Load training image
    image_index = ChosenTrainingImages(j);
    %%%%%%%%%%%%%%
    % Training the Neural Network
    
    % Calculate activation values with current netwrok (feedforward)
    [CurrentResults.activation,CurrentResults.weightedInput]=calculate_activation(trainingData.images(:,image_index),NeuralNetwork);
    % Set up label vector, and read the label for th current image
    CurrentResults.labelVector=zeros(size(CurrentResults.activation{NeuralNetwork.NLayer}));
    CurrentResults.labelVector(trainingData.labels(image_index)+1) = 1;
    % Backpropagate
    [CurrentResults.delta]=calculate_errors(CurrentResults,NeuralNetwork);
    % Correction on weights and biases
    [NeuralNetwork] =network_correction_step(CurrentResults,NeuralNetwork,GradientDescent);
  end
  % Test results
  if (mod(i,100)==0)
    disp(['Epoch: ',num2str(i)]);
    [effectiveness] = test_network(testData.images,testData.labels,NeuralNetwork);
    disp(['Effectiveness of network: ',num2str(effectiveness)]);
  end
end
toc;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Test current network
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [effectiveness] = test_network(images,labels,NeuralNetwork)
output = zeros(size(labels))-1; %init as -1, not a label
for i = 1:length(labels)
  [activation,~] = calculate_activation(images(:,i),NeuralNetwork);
  [~,output(i)] = max(activation{NeuralNetwork.NLayer});
  output(i) = output(i) - 1;
end
effectiveness = length(find((labels-output)==0))/length(labels);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MNIST
% Using the following code to read in MNIST:
% http://ufldl.stanford.edu/wiki/index.php/Using_the_MNIST_Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Image array
function images = load_MNIST_images(filename)
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);
numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');
images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);
fclose(fp);
% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;
end

% Label array
function labels = load_MNIST_labels(filename)
fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);
numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Mismatch in label count');
fclose(fp);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate activations and weights (feedforward)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [activation,WeightedInput] = calculate_activation(image,NeuralNetwork)
% Calculate activations and weighted imputs for all neurons in all layers
activation{1} = image(:);
WeightedInput{1} = [];
for i = 2:NeuralNetwork.NLayer
  WeightedInput{i} = NeuralNetwork.weight{i}*activation{i-1}+NeuralNetwork.bias{i};
  activation{i} = NeuralNetwork.ActivationFunction(WeightedInput{i});
end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate corrections to weights and biases using gradient descent (backpropagate)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [NeuralNetwork] = network_correction_step(CurrentResults,NeuralNetwork,GradientDescent)
for i = 2:NeuralNetwork.NLayer
  NeuralNetwork.bias{i}=NeuralNetwork.bias{i} - ...
    GradientDescent.stepSize*CurrentResults.delta{i}/GradientDescent.NTrainImages;
  NeuralNetwork.weight{i}=NeuralNetwork.weight{i}*(1-NeuralNetwork.RegularizationLambda*GradientDescent.stepSize) - ...
    GradientDescent.stepSize*(CurrentResults.delta{i}*CurrentResults.activation{i-1}')/GradientDescent.NTrainImages;
end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate error from curremt test (backpropagation)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [delta] = calculate_errors(CurrentResults,NeuralNetwork)
% Last layer
delta{NeuralNetwork.NLayer} = CurrentResults.activation{NeuralNetwork.NLayer}-CurrentResults.labelVector;  
% Backpropagate to previous layers
for i = (NeuralNetwork.NLayer-1):-1:2
  delta{i} = ...
    (NeuralNetwork.weight{i+1}'*delta{i+1}).*...
    NeuralNetwork.ActivationFunctionDeriv(CurrentResults.weightedInput{i});
end
end


