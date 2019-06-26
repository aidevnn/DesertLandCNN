using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;

namespace DesertLandCNN
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            NumDN.DebugNumPy = false;

            Console.WriteLine("Hello World! CNN");

            var raw = File.ReadAllLines("digits.csv").OrderBy(a => NumDN.GetRandom.NextDouble()).ToArray();
            Console.WriteLine($"raw data:{raw.Length}");
            (var dataX, var dataY) = importDataset(raw);

            (var trainX, var trainY, var testX, var testY) = dataSplitTraitest(dataX, dataY, 0.9);
            Console.WriteLine($"trainX Shape:{trainX.Shape.Glue()}");
            Console.WriteLine($"testX  Shape:{testX.Shape.Glue()}");
            Console.WriteLine($"trainY Shape:{trainY.Shape.Glue()}");
            Console.WriteLine($"testY  Shape:{testY.Shape.Glue()}");
            Console.WriteLine();

            var optimizer = new Adam<float>();
            var net = new Network<float>(optimizer, new CrossEntropy<float>(), (testX, testY));

            net.AddLayer(new Conv2d<float>(4, new int[] { 3, 3 }, new int[] { 1, 8, 8 }, 1, "same"));
            net.AddLayer(new ReLULayer<float>());
            //net.AddLayer(new TanhLayer<float>());
            net.AddLayer(new Dropout<float>(0.4));
            net.AddLayer(new BatchNormalization<float>());

            net.AddLayer(new Conv2d<float>(8, new int[] { 3, 3 }, 1, "same"));
            net.AddLayer(new ReLULayer<float>());
            //net.AddLayer(new TanhLayer<float>());
            net.AddLayer(new Dropout<float>(0.4));
            net.AddLayer(new BatchNormalization<float>());
            net.AddLayer(new FlattenLayer<float>());

            net.AddLayer(new DenseLayer<float>(64));
            net.AddLayer(new ReLULayer<float>());
            //net.AddLayer(new TanhLayer<float>());
            net.AddLayer(new Dropout<float>(0.25));
            net.AddLayer(new BatchNormalization<float>());

            net.AddLayer(new DenseLayer<float>(10));
            //net.AddLayer(new TanhLayer<float>());
            net.AddLayer(new SoftmaxLayer<float>());

            net.Summary();
            net.Fit(trainX, trainY, 10, 16);
        }

        static float[] toCategorical(int[] x)
        {
            var data = x.SelectMany(i => Enumerable.Range(0, 10).Select((v, k) => k == i ? 1f : 0f)).ToArray();
            return data;
        }

        static (float[], float[]) importDataset(string[] raw)
        {
            var data = raw.Select(l => l.Split(',').Select(float.Parse).ToArray()).ToArray();
            var dataX = data.SelectMany(l => l.Take(64)).ToArray();
            var dataY = toCategorical(data.Select(l => (int)l[64]).ToArray());

            return (dataX, dataY);
        }

        static (NDArray<float>, NDArray<float>, NDArray<float>, NDArray<float>) dataSplitTraitest(float[] dataX,float[] dataY, double t = 0.5)
        {
            int lX = dataX.Length / 64;
            int lY = dataY.Length / 10;
            int splitX = (int)(lX * t);
            int splitY = (int)(lY * t);
            float[] dataTrainX = dataX.Take(splitX * 64).ToArray();
            float[] dataTestX = dataX.Skip(splitX * 64).ToArray();
            float[] dataTrainY = dataY.Take(splitY * 10).ToArray();
            float[] dataTestY = dataY.Skip(splitY * 10).ToArray();

            var trainX = new NDArray<float>(dataTrainX, new int[] { splitX, 1, 8, 8 });
            var testX = new NDArray<float>(dataTestX, new int[] { lX - splitX, 1, 8, 8 });
            var trainY = new NDArray<float>(dataTrainY, new int[] { splitY, 10});
            var testY = new NDArray<float>(dataTestY, new int[] { lY - splitY, 10});

            return (trainX, trainY, testX, testY);
        }

        static void Test()
        {
            NumDN.DebugNumPy = true;

            //var a = NumDN.UniformInt(-10, 10, 2, 4, 3);
            //Console.WriteLine(a);

            //var b = NumDN.Sum(a, 1);
            //Console.WriteLine(b);

            //var c = NumDN.Sum(a, false, 1);
            //Console.WriteLine(c);

            //var d = NumDN.Sum(a, false);
            //Console.WriteLine(d);

            //int[] arr = { 8, 7, 2, 1, 0, 8, 8, 1, 0, 3, 4, 2, 7, 9, 0, 5, 3, 2, 3, 9, 2, 5, 1, 7 };
            //var e = new NDArray<int>(arr, new int[] { 2, 4, 3 });
            //Console.WriteLine(e);
            //Console.WriteLine();
            //Console.WriteLine(e.T);
            //Console.WriteLine();
            //Console.WriteLine(e.transpose(2, 0, 1));
            //Console.WriteLine();
            //Console.WriteLine(e.transpose(1, 2, 0));
            //Console.WriteLine();

            //var imgs = NumDN.UniformInt(0, 10, 2, 1, 4, 4);
            //Console.WriteLine(imgs);

            //var imgsPad = NumDN.Pad(imgs, (0, 0), (1, 1), (1, 2), (3, 4));
            //Console.WriteLine(imgsPad);


            //var i0 = NumDN.Repeat(NumDN.ARange(5), 2);
            //Console.WriteLine(i0.Glue());
            //i0 = NumDN.Tile(i0, 3);
            //Console.WriteLine(i0.Glue());

            //var pad = Conv2d<int>.determinePadding(new int[] { 3, 3 });
            //(var k, var i, var j) = Conv2d<int>.img2colIndices(new int[] { 4, 1, 4, 4 }, new int[] { 3, 3 }, 1, pad);
            //Console.WriteLine(k);
            //Console.WriteLine();
            //Console.WriteLine(i);
            //Console.WriteLine();
            //Console.WriteLine(j);

            //var a = new NDArray<int>(5, 3);
            //var b = new NDArray<int>(4, 1, 3);
            //var c = new NDArray<int>(2);

            //Console.WriteLine(NumDN.BroadCasting(a, b).Glue(globalformat:"({0})"));

            //var a = NumDN.UniformInt(0, 10, 3, 5, 4);
            //Console.WriteLine(NumDN.BroadCasting(a).Glue(globalformat:"({0})"));
            //a.GetIndexes(":");
            //a.GetIndexes(":2", "1:");
            //a.GetIndexes(":2", NumDN.NewAxis, "1:");
            //a.GetIndexes(NumDN.NewAxis, ":2", "1:");
            //a.GetIndexes(":2", "1:", NumDN.NewAxis);
            //a.GetIndexes(NumDN.NewAxis, ":2", "1:", NumDN.NewAxis);
            //a.GetIndexes(NumDN.NewAxis, ":2", "1:", NumDN.NewAxis, NumDN.NewAxis);
            //a.GetIndexes(NumDN.NewAxis, NumDN.NewAxis, ":2", "1:", NumDN.NewAxis);
            //a.GetIndexes(NumDN.NewAxis, ":2", NumDN.NewAxis, "1:", NumDN.NewAxis);
            //a.GetIndexes(NumDN.NewAxis, ":2", NumDN.NewAxis, NumDN.NewAxis, "1:", NumDN.NewAxis);
            //a.GetIndexes(NumDN.NewAxis, ":2", NumDN.NewAxis, NumDN.NewAxis);

            //var b = NumDN.UniformInt(0, 3, 2, 4, 3);
            //var c = NumDN.UniformInt(0, 5, 5, 4);
            ////var d = NumDN.UniformInt(0, 4, 2, 4);
            //Console.WriteLine(a);
            ////Console.WriteLine(b);
            //Console.WriteLine(c);
            //Console.WriteLine(a + c);
            //Console.WriteLine(NDArray<int>.Dot(a, c.T));
            ////Console.WriteLine(d);

            //var x = NumDN.UniformInt(0, 10, 4, 5, 3, 6);
            //var y = NumDN.UniformInt(0, 5, 3, 2);
            //var z = NumDN.UniformInt(0, 3, 3, 3, 2);
            //Console.WriteLine(x);
            //Console.WriteLine(y);
            //Console.WriteLine(z);

            //x.GetIndexes(":", ":2", NumDN.NewAxis, ":", "3:");
            //x.GetIndexes2(":", ":2", NumDN.NewAxis, ":", "3:");

            //x.GetIndexes(":", y);

            //x.GetIndexes(y, "1:3");
            //x.GetIndexes(y, z);
            //x.GetIndexes(":", z, ":1", z);
            //x.GetIndexes(":", z, z, ":1");
            //x.GetIndexes(":", ":1", z, z);
            //x.GetIndexes(y, NumDN.NewAxis, z);
            //x.GetIndexes(NumDN.NewAxis, y, z);

            //Console.WriteLine(a.T);
            //Console.WriteLine(NDArray<int>.Dot(a, b));

            //a.GetIndexes(":", c);
            //a.GetIndexes(b, "1:3");
            //a.GetIndexes(b, c, "2:3");
            //a.GetIndexes(c, ":", c);
            //a.GetIndexes(":", c, c);
            //a.GetIndexes(c, c, c);

            //var x = NumDN.UniformInt(0, 10, 3, 3);
            //var y = NumDN.UniformInt(0, 3, 2, 2);
            //var z = NumDN.UniformInt(0, 3, 2, 2);
            //Console.WriteLine(x);
            //Console.WriteLine(y);
            //Console.WriteLine(z);

            //var c = x.GetIndexes(":", z);
            //Console.WriteLine(c);

            //Console.WriteLine(ExtensionIndexes.AddAt(x, c).Indexes(":", y));

            //var x = NumDN.UniformInt(0, 16, 2, 1, 8, 8).Cast<double>();
            //Console.WriteLine(x);

            //var cols = Conv2d<double>.img2col(x, new int[] { 3, 3 }, 1);
            //Console.WriteLine(cols);

            //var x0 = Conv2d<double>.col2img(cols, new int[] { 2, 1, 8, 8 }, new int[] { 3, 3 }, 1);
            //Console.WriteLine(x0);

            //Bench(1000, 10);
            //Bench(1000, 10);
            //Bench(1000, 10);
            //Bench(1000, 10);
            //Bench(1000, 10);

            //var x = NumDN.UniformInt(0, 16, 3, 4, 5).Cast<double>();
            //Console.WriteLine(x);
            //Console.WriteLine(NumDN.Mean(x, 0));
            //Console.WriteLine(NumDN.Var(x, 0));
        }

        static void Bench(int batchSize, int loops)
        {
            NumDN.DebugNumPy = false;

            var sw = Stopwatch.StartNew();

            for (int k = 0; k < loops; ++k)
            {
                var x = NumDN.UniformInt(0, 16, batchSize, 1, 8, 8).Cast<float>();
                var cols = Conv2d<float>.img2col(x, new int[] { 3, 3 }, 1);
                var x0 = Conv2d<float>.col2img(cols, x.Shape, new int[] { 3, 3 }, 1);
            }

            Console.WriteLine($"{0.001 * sw.ElapsedMilliseconds} s");
        }
    }
}
