# load nist data from mat file
using MAT

file = matopen("nist_data.mat")

trainX = read(file, "X")
Y = read(file, "y")

trainY = zeros(size(trainX,1),10)
for i in 1:size(trainX,1)
  trainY[i,Int(Y[i])] = 1
end

testX = trainX
testY = trainY

close(file)
