# load mnist data from mat file
using MAT

function getY(rows, cols, column)
  y = zeros(rows,cols)
  y[:, column] = ones(rows,1)
  y
end

file = matopen("mnist_all.mat")

trainX = read(file, "train0")
trainY = getY(size(trainX,1), 10, 10)

testX = read(file, "test0")
testY = getY(size(testX,1), 10, 10)

for i in 1:9
  tmp = read(file,string("train",i))
  trainX = [trainX; tmp]
  trainY = [trainY; getY(size(tmp,1),10,i)]

  tmp = read(file,string("test",i))
  testX = [testX; tmp]
  testY = [testY; getY(size(tmp,1),10,i)]
end

close(file)
