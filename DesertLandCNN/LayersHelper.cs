using System;
using System.Collections.Generic;
using System.Linq;

namespace DesertLandCNN
{

    public abstract class Layer<Type>
    {
        public string Name;
        public bool trainable = true;
        public int[] Inputs { get; set; }
        public int[] Outputs { get; set; }
        public abstract void SetInputShape(int[] inputShape);
        public abstract int[] GetOutputShape();
        public abstract int Parameters { get; }
        public abstract void Initialize(IOptimizer<Type> optimizer = null);
        public abstract NDArray<Type> Forward(NDArray<Type> X, bool isTraining);
        public abstract NDArray<Type> Backward(NDArray<Type> accumGrad);
    }

    public class ActivationLayer<Type> : Layer<Type>
    {
        static Dictionary<string, IActivation<Type>> Activations;

        static ActivationLayer()
        {
            Activations = new Dictionary<string, IActivation<Type>>();
            Activations["identity"] = new IdentityActivation<Type>() as IActivation<Type>;
            Activations["sigmoid"] = new SigmoidActivation<Type>() as IActivation<Type>;
            Activations["tanh"] = new TanhActivation<Type>() as IActivation<Type>;
            Activations["relu"] = new ReLUActivation<Type>() as IActivation<Type>;
            Activations["softmax"] = new SoftmaxActivation<Type>() as IActivation<Type>;
        }

        private readonly IActivation<Type> activation;
        private NDArray<Type> layerInput;

        public override int Parameters => 0;

        public ActivationLayer(string name)
        {
            activation = Activations[name];
            Name = name.ToUpper();
        }

        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            return accumGrad * activation.Gradient(layerInput);
        }

        public override void Initialize(IOptimizer<Type> optimizer = null) { }

        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining)
        {
            layerInput = new NDArray<Type>(X);
            return activation.Function(X);
        }

        public override void SetInputShape(int[] inputShape)
        {
            Inputs = inputShape.ToArray();
            Outputs = inputShape.ToArray();
        }

        public override int[] GetOutputShape() => Outputs;
    }

    public class IdentityLayer<Type> : ActivationLayer<Type>
    {
        public IdentityLayer() : base("identity") { }
    }

    public class SigmoidLayer<Type> : ActivationLayer<Type>
    {
        public SigmoidLayer() : base("sigmoid") { }
    }

    public class TanhLayer<Type> : ActivationLayer<Type>
    {
        public TanhLayer() : base("tanh") { }
    }

    public class ReLULayer<Type> : ActivationLayer<Type>
    {
        public ReLULayer() : base("relu") { }
    }

    public class SoftmaxLayer<Type> : ActivationLayer<Type>
    {
        public SoftmaxLayer() : base("softmax") { }
    }

    public class DenseLayer<Type> : Layer<Type>
    {
        private NDArray<Type> W, w0, layerInput;
        private IOptimizer<Type> WOpt, w0Opt;

        public override int Parameters => NumDN.ShapeLength(W.Shape) + NumDN.ShapeLength(w0.Shape);

        public override int[] GetOutputShape() => Outputs;

        public DenseLayer(int nodes, int[] inputShape = null)
        {
            Outputs =new int[] { nodes };
            Name = "Dense";
            if (inputShape != null)
                Inputs = inputShape.ToArray();
        }

        public override void Initialize(IOptimizer<Type> optimizer)
        {
            double lim = 1.0 / Math.Sqrt(Inputs[0]);

            W = NumDN.Uniform<Type>(-lim, lim, Inputs[0], Outputs[0]);
            w0 = NDArray<Type>.Zeros(1, Outputs[0]);

            WOpt = optimizer.Clone();
            w0Opt = optimizer.Clone();
        }

        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining = true)
        {
            layerInput = new NDArray<Type>(X);
            return NDArray<Type>.Dot(X, W) + w0;
        }

        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            var Wtmp = new NDArray<Type>(W);

            if (trainable)
            {
                var gW = NDArray<Type>.Dot(layerInput.T, accumGrad);
                var gw0 = (new NDArray<Type>(accumGrad.Shape)) + NumDN.SumDouble(accumGrad);

                W = WOpt.Update(W, gW);
                w0 = w0Opt.Update(w0, gw0);
            }

            var accumGrad0 = NDArray<Type>.Dot(accumGrad, W.T);
            return accumGrad0;
        }

        public override void SetInputShape(int[] inputShape)
        {
            Inputs = inputShape.ToArray();
        }
    }

    public class FlattenLayer<Type> : Layer<Type>
    {
        int[] prevShape;

        public override int[] GetOutputShape() => new int[] { NumDN.ShapeLength(Inputs) };
        public FlattenLayer(int[] inputShape = null)
        {
            Name = "Flatten";
            if (inputShape != null)
                Inputs = inputShape.ToArray();
        }

        public override int Parameters => NumDN.ShapeLength(Inputs);

        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            return accumGrad.ReShape(prevShape);
        }

        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining)
        {
            prevShape = X.Shape.ToArray();
            return X.ReShape(X.Shape[0], -1);
        }

        public override void Initialize(IOptimizer<Type> optimizer = null)
        {

        }

        public override void SetInputShape(int[] inputShape)
        {
            Inputs = inputShape.ToArray();
        }
    }

    public class Dropout<Type> : Layer<Type>
    {
        double p;
        bool[] mask;
        public override int[] GetOutputShape() => Inputs;
        public Dropout(double p = 0.2)
        {
            this.p = p;
            Name = "Dropout";
        }

        public override int Parameters => 0;

        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            return accumGrad.Apply((v, i) => mask[i % mask.Length] ? v : NDArray<Type>.OpsT.Zero);
        }

        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining)
        {
            mask = X.items.Select(x => NumDN.GetRandom.NextDouble() > p).ToArray();
            return X.Apply((v, i) => mask[i] ? v : NDArray<Type>.OpsT.Zero);
        }

        public override void Initialize(IOptimizer<Type> optimizer = null)
        {

        }

        public override void SetInputShape(int[] inputShape)
        {
            Inputs = inputShape.ToArray();
        }
    }

    public class BatchNormalization<Type> : Layer<Type>
    {
        double momentum;
        double eps = 0.01;
        NDArray<Type> gamma, beta, runningMean, runningVar, X_centered, stddev_inv;
        IOptimizer<Type> gOpt, bOpt;

        public override int[] GetOutputShape() => Inputs;

        public BatchNormalization(double momentum = 0.99)
        {
            this.momentum = momentum;
            Name = "BatchNormalization";
        }

        public override int Parameters => NumDN.ShapeLength(gamma.Shape) + NumDN.ShapeLength(beta.Shape);

        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            var gamma0 = new NDArray<Type>(gamma);
            if (trainable)
            {
                var X_norm = X_centered * stddev_inv;
                var grad_gamma = NumDN.Sum(accumGrad * X_norm, 0);
                var grad_beta = NumDN.Sum(accumGrad, 0);

                gamma = gOpt.Update(gamma, grad_gamma);
                beta = bOpt.Update(beta, grad_beta);
            }

            double batchSize = accumGrad.Shape[0];

            accumGrad = (1.0 / batchSize) * gamma0 * stddev_inv * (
                batchSize * accumGrad
                - NumDN.Sum(accumGrad, 0)
                - X_centered * stddev_inv * stddev_inv * NumDN.Sum(accumGrad * X_centered, 0)
                );

            return accumGrad;
        }

        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining)
        {
            if (runningMean == null)
            {
                runningMean = NumDN.Mean(X, 0);
                runningVar = NumDN.Var(X, 0);
            }

            NDArray<Type> mean, varr;
            if (isTraining)
            {
                mean = NumDN.Mean(X, 0);
                varr = NumDN.Var(X, 0);

                runningMean = momentum * runningMean + (1 - momentum) * mean;
                runningVar = momentum * runningVar + (1 - momentum) * varr;
            }
            else
            {
                mean = runningMean;
                varr = runningVar;
            }

            X_centered = X - mean;
            stddev_inv = 1.0 / NumDN.Sq(varr + eps);

            var X_norm = X_centered * stddev_inv;
            var output = gamma * X_norm + beta;

            return output;
        }

        public override void Initialize(IOptimizer<Type> optimizer = null)
        {
            gamma = NDArray<Type>.Ones(Inputs);
            beta = NDArray<Type>.Zeros(Inputs);
            gOpt = optimizer.Clone();
            bOpt = optimizer.Clone();
        }

        public override void SetInputShape(int[] inputShape)
        {
            Inputs = inputShape.ToArray();
        }
    }
}
