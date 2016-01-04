println("################################################")
println("##   BackpropNeuralNetwork MNIST Demo")
println("##   by Imanol Schlag, 2015")
println("################################################")

println("\n>loading libraries... (press enter)")
readline(STDIN);
include("BackpropNeuralNetwork.jl")
using MAT
using Images, ImageView

## MINST number recognition
println("\n>loading MNIST data... (press enter)")
readline(STDIN);
include("LoadMNIST.jl")
println("trainX: ", size(trainX))
println("trainY: ", size(trainY))
println("testX: ", size(testX))
println("testY: ", size(testY))

# visualize 25 random numbers
println("\n>visualizing 25 random test samples ... (press enter)")
readline(STDIN);
rows = shuffle(collect(1:size(testX,1)))[1:25]
r = Int(sqrt(size(testX,2)))
I = reshape(testX[rows[1],:],r,r)
for i in collect(2:25)
  I = [I; reshape(testX[rows[i],:],r,r)]
end
view(grayim(I))

# create net
println("\n>initializing new neural network ... (press enter)")
readline(STDIN);
layers = [784 100 10]
net = FeedForwardNetwork(layers)
println("layers: ", layers)

# evaluate before training
println("\n>evaluate untrained network on train data... (press enter)")
readline(STDIN);
evaluate(trainX,trainY,net)
println("\n>evaluate untrained network on test data... (press enter)")
readline(STDIN);
evaluate(testX,testY,net)

# train network
println("\n>train neural network with backprop... ")
epochs = 500
η = 0.5
ω = 100
println("epochs: $epochs")
println("learnign rate: $η")
println("mini batch size: $ω")
println("(press enter)")
readline(STDIN);
train(η, epochs, ω, trainX, trainY, [], [], net)

# evaluate after training
println("\n>evaluate trained network on train data... (press enter)")
readline(STDIN);
evaluate(trainX,trainY,net)

println("\n>evaluate trained network on test data... (press enter)")
readline(STDIN);
evaluate(testX,testY,net)
