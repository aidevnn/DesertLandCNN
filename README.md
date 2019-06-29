# DesertLandCNN

The dataset is "digits.cs" from scikit-learn.

Building Layers
```

var optimizer = new Adam<float>();
var net = new Network<float>(optimizer, new CrossEntropy<float>(), (testX, testY));

net.AddLayer(new Conv2d<float>(4, new int[] { 3, 3 }, new int[] { 1, 8, 8 }, 1, "same"));
net.AddLayer(new ReLULayer<float>());
net.AddLayer(new Dropout<float>(0.4));
net.AddLayer(new BatchNormalization<float>());

net.AddLayer(new Conv2d<float>(8, new int[] { 3, 3 }, 1, "same"));
net.AddLayer(new ReLULayer<float>());
net.AddLayer(new Dropout<float>(0.4));
net.AddLayer(new BatchNormalization<float>());
net.AddLayer(new FlattenLayer<float>());

net.AddLayer(new DenseLayer<float>(64));
net.AddLayer(new ReLULayer<float>());
net.AddLayer(new Dropout<float>(0.25));
net.AddLayer(new BatchNormalization<float>());

net.AddLayer(new DenseLayer<float>(10));
net.AddLayer(new SoftmaxLayer<float>());

net.Summary();
net.Fit(trainX, trainY, 10, 16);
```

A poor but quick test. The actual code run very slowly. A futur update will introduce the Stride and the Storage of NDarray for more performance.

### Output

```
Hello World! CNN
raw data:1797
trainX Shape:1617 1 8 8
testX  Shape:180 1 8 8
trainY Shape:1617 10
testY  Shape:180 10

Summary
Input Shape:System.Int32[]
Layer: Conv2d     Parameters:  40 Nodes[In:(1 8 8) -> Out:(4 8 8)]
Layer: RELU       Parameters:   0 Nodes[In:(4 8 8) -> Out:(4 8 8)]
Layer: Dropout    Parameters:   0 Nodes[In:(4 8 8) -> Out:(4 8 8)]
Layer: BatchNormalization Parameters: 512 Nodes[In:(4 8 8) -> Out:(4 8 8)]
Layer: Conv2d     Parameters: 296 Nodes[In:(4 8 8) -> Out:(8 8 8)]
Layer: RELU       Parameters:   0 Nodes[In:(8 8 8) -> Out:(8 8 8)]
Layer: Dropout    Parameters:   0 Nodes[In:(8 8 8) -> Out:(8 8 8)]
Layer: BatchNormalization Parameters: 1024 Nodes[In:(8 8 8) -> Out:(8 8 8)]
Layer: Flatten    Parameters: 512 Nodes[In:(8 8 8) -> Out:(512)]
Layer: Dense      Parameters: 32832 Nodes[In:(512) -> Out:(64)]
Layer: RELU       Parameters:   0 Nodes[In:(64) -> Out:(64)]
Layer: Dropout    Parameters:   0 Nodes[In:(64) -> Out:(64)]
Layer: BatchNormalization Parameters: 128 Nodes[In:(64) -> Out:(64)]
Layer: Dense      Parameters: 650 Nodes[In:(64) -> Out:(10)]
Layer: SOFTMAX    Parameters:   0 Nodes[In:(10) -> Out:(10)]
Output Shape:(10)
Total Parameters:35994

Start Training...
Progress ####################
Epochs     0/10 Loss:0.694021 Acc:0.0006
Progress ####################
Epochs     1/10 Loss:0.652547 Acc:0.0062
Progress ####################
Epochs     2/10 Loss:0.715940 Acc:0.0136
Progress ####################
Epochs     3/10 Loss:0.757860 Acc:0.0136
Progress ####################
Epochs     4/10 Loss:0.858162 Acc:0.0217
Progress ####################
Epochs     5/10 Loss:0.885194 Acc:0.0167
Progress ####################
Epochs     6/10 Loss:1.026447 Acc:0.0241
Progress ####################
Epochs     7/10 Loss:1.091511 Acc:0.0241
Progress ####################
Epochs     8/10 Loss:1.148595 Acc:0.0285
Progress ####################
Epochs     9/10 Loss:1.218037 Acc:0.0291
Progress ####################
Epochs    10/10 Loss:1.335522 Acc:0.0303
End Training.00:02:50.1677183
Validation Loss:1.113299 Acc:0.0455
```


#### References.
Base code for layers / activations / network was in python and comes from this very great and useful ML repo https://github.com/eriklindernoren/ML-From-Scratch
