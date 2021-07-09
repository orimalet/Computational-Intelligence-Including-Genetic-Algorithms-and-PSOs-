# Computational-Intelligence-Including-Genetic-Algorithms-and-PSOs-

This coursework is made to assess and enhance your understanding on various mechanisms on neural
network learning and evolutionary optimisation and specifically, basic neural architectures, their learning
procedures, function optimisation and evolutionary search operators. The coursework is configured as a
multi-stage mini-project where you are asked to: design, implement, experiment and discuss various
components. You can choose any single (just one please!) implementation platform of your choice, but either
Matlab, Python or R is recommended as it is far easier to code in these high level programming languages.
For Parts 1 & 2 it is straightforward how to implement the underlying equations from scratch without using
any additional neural network libraries, but for Part 3 feel free to make use of any toolboxes or libraries or
packages supported by your implementation platform; this is also clarified below.


Questions

Part 1: [5 points]
Use the basic equations for its training from our notes, to implement a simple Perceptron for a given
classification problem. The number of inputs and the classification problem to solve will be specifiable by the
user. Implement this from scratch using, for example, for-loops to iterate around epochs and patterns as in
the lecture notes. Try to make the code as simple as possible by using arrays/matrices/vectors. The training
should carry on until some maximum number of epochs is reached or until all patterns are learned. Include
the design for this stage (that is basic equations and structure of the algorithm), the code, and also some
experiments for 2-3 low- or high-dimensional example problems of your choice (you can download a dataset
from the internet, or define one directly, or generate one easily yourselves). Display the weight adaptations
during the epochs to show the learning process. Extra marks if your program displays the boundary (for 2D
or 3D only!).


Part 2: [10 points]
Use the basic equations of Backpropagation from our notes to implement and simulate a simple MLP network.
The architecture (number of inputs, number of layers and number of nodes per layer, and number of outputs)
can be given by the user or be fixed inside your program. Implement both the forward and the backward 
passes separately, and use a simple activation function (e.g., logistic, relu, etc.). Include the design for this
stage (that is basic equations, structure and sequencing of operations), the code, and also some experiments
for 2-3 simple problems of your choice (e.g., toy problems (XOR, symmetry checking), or using random points
from a multi-dimensional function you randomly create). Discuss the learning ability in terms of reducing the
fitting error in your patterns and experiment with different architectures (you can ignore generalisation
issues and just focus on weight adaptation only). Extra marks will be given for momentum incorporation, or
experimentation with different activation functions, or use complexity control.


Part 3: [15 points]
This part requires for the student to obtain a little bit of self-familiarisation with some existing standard
libraries (just to avoid implementing the evolutionary optimisers from scratch). Therefore, use a
toolbox/library/package for the platform of your choice (various options are given below) to create a genetic
algorithm (GA) and a particle swarm optimisation (PSO) to train the multilayer neural network from the
previous Part 2. The GA and PSO optimisers should search for the best weights for the network to solve a
simple problem (such as one from Part 2, or any different classification/regression problem of your choice).
Only the forward pass and the overall output error need to be evaluated for the objective function of the GA
and the PSO. For the GA, you can use any type of chromosome encoding you like (binary or real-valued, or
both!) and any fitness scaling procedure (scaling/ranking), or genetic crossover/mutation operator you think
is the most efficient for your problem. Try to briefly justify your decisions for choosing the specific
mechanisms or operators. Compare a few different options your library supports (e.g., different
population/swarm sizes, generations, mutation/crossover rates, stopping criteria or whatever is supported
by the implementation you use). Include a design section that explains how you designed your
evolutionary-based MLP training, its main components, alternatives, and how you tested them. Extra marks
will be given for experimentations with different GA and PSO operators. You can also experiment optionally
with any other evolutionary optimisation method (other than PSO or GA) your library supports. Feel free to
use whatever your libraries support, but please, explain what they do J
