
# Normalizes the features in x
# The mean value of each feature is 0 and the standard deviation is 1
# Returns normalized x, μ and σ
function featureNormalize(x)
rows = size(x,1)
cols = size(x,2)

μ = mean(x,1)
σ = std(x,1)
xNorm = zeros(x)

# normalize
for i in 1:cols
	for j in 1:rows
		xNorm[j,i] = (x[j,i] - μ[i]) / σ[i];
	end
end

(xNorm, μ, σ)
end
