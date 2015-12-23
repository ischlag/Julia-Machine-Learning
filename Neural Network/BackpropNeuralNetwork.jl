#=
  Backprop Neural Network
    - sigmoid neurons
    - actiavtion level: Σ(w*a + b)

  by Imanol Schlag

  Usage:
  1.) Init network.
      layers = [2 3 1]
      means 2 inputs, 3 hidden and 1 output neuron

=#
type FeedForwardNetwork
  layers::Array{Int64}
  # weights
  W::Array{Array{Float64}}
  # biases
  B::Array{Array{Float64}}

  function FeedForwardNetwork(layers::Array{Int64})
    this = new()
    this.layers = layers
    this.W, this.B = randomInit(this.layers)
    this
  end
end

# Randomly initialize the weights and biases
function randomInit(layers::Array{Int64})
  max = length(layers)
  weights = Array{Array{Float64}}(max-1)
  biases = Array{Array{Float64}}(max-1)
  for i in 1:max-1
    weights[i] = randn(layers[i], layers[i+1])
    biases[i] = randn(1, layers[i+1])
  end
  (weights, biases)
end

# activation function and its first derivative
sigmoid(z) = 1 ./ (1 + exp(-z))
sigmoidPrime(z) = sigmoid(z) .* (1-sigmoid(z))
#sigmoidPrime(z) = exp(-z)/((1+exp(-z))^2)

# Forward propagation
function forward(x, net::FeedForwardNetwork)
  println("Network: ", net.layers)
  if(size(x,2) != net.layers[1])
    error("x doesn't match input layer (", size(x,2)," vs ", net.layers[1],")")
  end

  for i in 1:size(net.W,1)
    #println(i, ": ",size(x), " * ", size(net.W[i]))
    x = x * net.W[i] .+ net.B[i] # sum activation for all samples
    x = sigmoid(x)
  end
  x
end

# calculate the cost given the current weights and biases
# C(W,B) = 1/2n Σ ||y-a||^2
function cost(x, y, net::FeedForwardNetwork)
  a = forward(x, net)
  n = size(x,1)
  1/2n * sum((y-a).^2)
end

# find the gradients for one sample
#=
for [2 3 1]
σ := activation function (i.e. sigmoid)
a1 = x
a1 * W1 + B1 = z2  (+ oerator is not matrix plus!)
σ(z2) = a2
a2 * W2 + B2 = z3
σ(z3) = a3
y - a3 = q
1/2 * q^2 = C(W,B)   (cost is the quadratic error)

Example for gradient with respect to W1
∂C(W,B)/∂W1 = ∂C/∂q * ∂q/∂a3 * ∂a3/∂z3 * ∂z3/∂a2 * ∂a2/∂z2 * ∂z2/∂W1
∂C(W,B)/∂W1 =   q   *  -1    * σ'(z3)  *   W2    *  σ'(z2) * a1
∂C(W,B)/∂W1 =    (a3 - y)    * σ'(z3)  *   W2    *  σ'(z2) * a1
∂C(W,B)/∂W1 =              δ3          *   W2    *  σ'(z2) * a1
∂C(W,B)/∂W1 =                          δ2                  * a1

=#
function backprop(x, y, net)
  max = length(net.layers)

  # forward saving a and z
  a = Array{Array{Float64}}(max)
  z = Array{Array{Float64}}(max)
  a[1] = x
  for i in 1:max-1
    z[i+1] = a[i] * net.W[i] .+ net.B[i]
    a[i+1] = sigmoid(z[i+1])
  end

  # calculate δ error through backprop
  δ = Array{Array{Float64}}(max)
  δ[max] = (a[max] - y) .* sigmoidPrime(z[max]) # error of output level
  for i in reverse(2:max-1)
    δ[i] = δ[i+1] * net.W[i]' .* sigmoidPrime(z[i])
  end

  # calculate gradients for all W and b
  ΔW = Array{Array{Float64}}(max-1)
  ΔB = Array{Array{Float64}}(max-1)
  for i in 1:max-1
    ΔW[i] = a[i]' * δ[i+1]
    ΔB[i] = δ[i+1]
  end
  (ΔW, ΔB)
end

#=
  X   rows are samples, columns is input dimensions
  y   rows are samples, columns is output dimensions

  performs one step based on the average gradient over all given samples in X
  given only the 2 dimensions W1 and W2
  ΔC := the cost
  ∇C := vector of partial derivatives of cost
  η := learning rate (positive and small)

  ΔC ≈ ∂C/∂W1 * v1 + ∂C/∂W2 * v2

  [v1 v2]' := vector of change, Δv
  [∂C/∂W1 ∂C/∂W2]' := vector of partial derivatives, ∇C

  ΔC = ∇C * Δv (rewritten)
  Δv = -η∇C

  now we can make a step (substitute)
  ΔC ≈ ∇C * -η∇C
  ΔC ≈ -η(∇C)^2
  therefore: ΔC ⋜ 0 (which means that our cost decreases)

=#
function gradientDescentBatch(X, Y, learningRate, net::FeedForwardNetwork)
  η = learningRate
  ∇W = Array{Array{Float64}}(length(net.layers)-1)
  ∇B = Array{Array{Float64}}(length(net.layers)-1)
  for i = 1:length(net.layers)-1
    ∇W[i] = zeros(net.layers[i], net.layers[i+1])
    ∇B[i] = zeros(1,net.layers[i+1])
  end
  count = size(X,1)
  for i = 1:count
    w, b = backprop(X[i,:],Y[i,:], net)
    ∇W += w
    ∇B += b
  end

  net.W -= (η/count * ∇W)
  net.B -= (η/count * ∇B)

end

function evaluate(X,Y,net)
  if net.layers[end] == 1
    # regression
    # TODO
  else
    # classification
    res = forward(X,net)
    println("res: ", res)
    count = 0
    for i = 1:size(res,1)
      r = res[i,:]
      c = indmax(r[1,:])
      if Y[i,c] == 1
        println(i, ",",c," == 1")
        count += 1
      end
    end
  end
  p = count/size(res,1)
  p
end
