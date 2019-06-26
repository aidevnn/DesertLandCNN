using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace DesertLandCNN
{
    public class Network<Type>
    {
        public readonly IOptimizer<Type> optimizer;
        public readonly ILoss<Type> lossFunction;
        public List<Layer<Type>> layers = new List<Layer<Type>>();

        NDArray<Type> testX, testY;

        public Network(IOptimizer<Type> optimizer, ILoss<Type> loss,(NDArray<Type>, NDArray<Type>) tests)
        {
            this.optimizer = optimizer;
            lossFunction = loss;
            (testX, testY) = tests;
        }

        public void SetTrainable() => layers.ForEach(l => l.trainable = true);

        public void SetNonTrainable() => layers.ForEach(l => l.trainable = false);

        public void AddLayer(Layer<Type> layer)
        {
            if (layers.Count != 0)
                layer.SetInputShape(layers.Last().GetOutputShape());

            layer.Initialize(optimizer);
            layers.Add(layer);
        }

        public NDArray<Type> ForwardPass(NDArray<Type> X, bool isTraining = true)
        {
            var layerOutput = new NDArray<Type>(X);
            foreach (var layer in layers)
                layerOutput = layer.Forward(layerOutput, isTraining);

            return layerOutput;
        }

        public void BackwardPass(NDArray<Type> lossGrad)
        {
            foreach (var layer in layers.Reverse<Layer<Type>>())
                lossGrad = layer.Backward(lossGrad);
        }

        public NDArray<Type> Predict(NDArray<Type> X) => ForwardPass(X, false);

        (double, double) TestOnBatch(NDArray<Type> X, NDArray<Type> y)
        {
            var yp = ForwardPass(X, false);
            var loss = NumDN.MeanDouble(lossFunction.Loss(y, yp));
            var acc = lossFunction.Acc(y, yp);
            return (loss, acc);
        }

        public (double, double) TrainOnBatch(NDArray<Type> X, NDArray<Type> y)
        {
            var yp = ForwardPass(X);
            var loss = NumDN.MeanDouble(lossFunction.Loss(y, yp));
            var acc = lossFunction.Acc(y, yp);
            var lossGrad = lossFunction.Grad(y, yp);
            BackwardPass(lossGrad);

            return (loss, acc);
        }

        public void Fit(NDArray<Type> X, NDArray<Type> y, int epochs, int batchSize = 64, int displayEpochs = 1)
        {
            var sw = Stopwatch.StartNew();
            Console.WriteLine("Start Training...");

            double step = 20.0 * batchSize / X.Shape[0];

            SetTrainable();
            List<double> losses = new List<double>();
            List<double> accs = new List<double>();
            for (int k = 0; k <= epochs; ++k)
            {
                var batchDataTrain = BatchIterator(X, y, batchSize);
                losses.Clear();
                accs.Clear();
                int cur = 0;
                double total = 0;
                double count = 0;
                Console.Write("Progress ");
                foreach (var batch in batchDataTrain)
                {
                    var (loss, acc) = TrainOnBatch(batch.Item1, batch.Item2);
                    losses.Add(loss);
                    accs.Add(acc);
                    int ntotal = (int)Math.Round(total);
                    if (ntotal > cur)
                    {
                        int diff = ntotal - cur + 1;
                        Console.Write(Enumerable.Repeat("#", diff).Glue(""));
                        cur += diff;
                    }
                    total += step;
                    count += batch.Item1.Shape[0];
                }

                Console.WriteLine(Enumerable.Repeat("#", 20 - Math.Min(20, cur)).Glue(""));
                if (k % displayEpochs == 0)
                    Console.WriteLine("Epochs {0,5}/{1} Loss:{2:0.000000} Acc:{3:0.0000}", k, epochs, losses.Average(), accs.Sum() / count);
            }
            Console.WriteLine($"End Training.{sw.Elapsed}");

            SetNonTrainable();
            var batchDataTest = BatchIterator(testX, testY, batchSize);
            losses.Clear();
            accs.Clear();
            double count1 = 0;
            foreach (var batch in batchDataTest)
            {
                var (loss, acc) = TestOnBatch(batch.Item1, batch.Item2);
                losses.Add(loss);
                accs.Add(acc);
                count1 += batch.Item1.Shape[0];
            }

            Console.WriteLine($"Validation Loss:{losses.Average():0.000000} Acc:{accs.Sum() / count1:0.0000}");
        }

        public void Summary()
        {
            Console.WriteLine("Summary");
            Console.WriteLine($"Input Shape:{layers[0].Inputs}");
            int tot = 0;
            foreach (var layer in layers)
            {
                Console.WriteLine($"Layer: {layer.Name,-10} Parameters: {layer.Parameters,3} Nodes[In:({layer.Inputs.Glue()}) -> Out:({layer.GetOutputShape().Glue()})]");
                tot += layer.Parameters;
            }

            Console.WriteLine($"Output Shape:({layers.Last().GetOutputShape().Glue()})");
            Console.WriteLine($"Total Parameters:{tot}");
            Console.WriteLine();
        }

        public static List<(NDArray<Type>, NDArray<Type>)> BatchIterator(NDArray<Type> X, NDArray<Type> y, int batchSize)
        {
            int nbSamples = X.Shape[0];
            var shapeX = X.Shape.ToArray();
            var shapeY = y.Shape.ToArray();
            int nb = nbSamples / Math.Min(nbSamples, batchSize);
            var rg = Enumerable.Range(0, nb).Select(i => i * batchSize).ToList();
            List<(NDArray<Type>, NDArray<Type>)> data = new List<(NDArray<Type>, NDArray<Type>)>();

            foreach (var i in rg)
            {
                var (begin, end) = (i, Math.Min(i + batchSize, nbSamples));
                shapeX[0] = batchSize;
                shapeY[0] = batchSize;
                int diff = end - begin;

                int[] rng = Enumerable.Range(0, batchSize).OrderBy(a => NumDN.GetRandom.NextDouble()).ToArray();

                NDArray<Type> X0 = NDArray<Type>.Zeros(shapeX);
                NDArray<Type> y0 = NDArray<Type>.Zeros(shapeY);
                for (int n = 0; n < batchSize; ++n)
                {
                    int m = rng[n];
                    X0[m] = X[begin + m % diff];
                    y0[m] = y[begin + m % diff];
                }

                data.Add((X0, y0));
            }

            return data;
        }
    }
}
