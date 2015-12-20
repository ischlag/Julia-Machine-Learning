# Multi Variable Linear Regression
# Inspired by the Machine Learning Course at Stanford
# Imanol

using Gadfly

include("featureNormalize.jl")

println("Loading data ... ")
data = readdlm("data2.txt",',')
x = data[:,1:2]
y = data[:, 3]
m = length(y)

@printf("First 10 examples from the dataset: \n");
t = [x[1:10,:] y[1:10,:]]'
for i in 1:10
  @printf(" x = [%.0f %.0f], y = %.0f \n", t[1,i], t[2,i], t[3,i]);
end

# Scale features and set them to zero mean
(x, μ, σ) = featureNormalize(x);

# Add intercept term to x
x = [ones(m,1) x]

#### Run Gradient Descent
α = 0.001
numIter = 4000
θ = zeros(3,1)
jHist = zeros(numIter, 1)

for i in 1:numIter
  # next theta
  θ = θ - (α/m) * (x' * ((x*θ)-y))
  # compute cost
  jHist[i] = sum((x*θ-y).^2)/(2m)
end

# plot convergence graph
pl = plot(
  x=collect(1:numIter),
  y=jHist,
  Guide.xlabel("Iterations"),
  Guide.ylabel("Error"),
  Guide.title("Convergence Graph"),
  Geom.line
  )
draw(SVGJS("jHist.js.svg", 6inch, 6inch), pl)

# Estimate the price of a 1650 sq-ft, 3 br house
price = [1, (1650-μ[1])/σ[1], (3-μ[2])/σ[2]]' * θ
println("Estimated price for a 1650 sq-ft, 3 br house: $price")

println("done!")
