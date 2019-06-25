using System;
using System.Linq;

namespace DesertLandCNN
{
    public interface IActivation<Type>
    {
        NDArray<Type> Function(NDArray<Type> x);
        NDArray<Type> Gradient(NDArray<Type> x);
    }

    public interface ILoss<Type>
    {
        NDArray<Type> Loss(NDArray<Type> y, NDArray<Type> p);
        NDArray<Type> Grad(NDArray<Type> y, NDArray<Type> p);
        double Acc(NDArray<Type> y, NDArray<Type> p);
    }

    public interface IOptimizer<Type>
    {
        NDArray<Type> Update(NDArray<Type> w, NDArray<Type> g);
        IOptimizer<Type> Clone();
    }

    public class IdentityActivation<Type> : IActivation<Type>
    {
        public NDArray<Type> Function(NDArray<Type> x) => x;

        public NDArray<Type> Gradient(NDArray<Type> x) => NDArray<Type>.Ones(x.Shape);
    }

    public class SigmoidActivation<Type> : IActivation<Type>
    {
        public NDArray<Type> Function(NDArray<Type> x) => NumDN.Sigmoid(x);

        public NDArray<Type> Gradient(NDArray<Type> x) => NumDN.Sigmoid(x) * (1 - NumDN.Sigmoid(x));
    }

    public class TanhActivation<Type> : IActivation<Type>
    {
        public NDArray<Type> Function(NDArray<Type> x) => NumDN.Tanh(x);

        public NDArray<Type> Gradient(NDArray<Type> x) => 1 - NumDN.Sq(NumDN.Tanh(x));
    }

    public class SoftmaxActivation<Type> : IActivation<Type>
    {
        public NDArray<Type> Function(NDArray<Type> x)
        {
            var ex = NumDN.Exp(x - NumDN.Max(x));
            return ex / NumDN.SumDouble(ex);
        }

        public NDArray<Type> Gradient(NDArray<Type> x)
        {
            var p = Function(x);
            return p * (1 - p);
        }
    }

    public class SquareLoss<Type> : ILoss<Type>
    {
        public double Acc(NDArray<Type> y, NDArray<Type> p) => 0;

        public NDArray<Type> Grad(NDArray<Type> y, NDArray<Type> p) => -(y - p);

        public NDArray<Type> Loss(NDArray<Type> y, NDArray<Type> p) => 0.5 * NumDN.Sq(y - p);
    }

    public class CrossEntropy<Type> : ILoss<Type>
    {
        public double Acc(NDArray<Type> y, NDArray<Type> p)
        {
            var m = (y - p).items.Select(x => Math.Abs(Convert.ToDouble(x)) < 0.5 ? 1.0 : 0.0).Average();
            return m;
        }

        public NDArray<Type> Grad(NDArray<Type> y, NDArray<Type> p)
        {
            var p0 = NumDN.Clamp(p, 1e-7, 1 - 1e-7);
            return -y / p0 + (1 - y) / (1 - p0);
        }

        public NDArray<Type> Loss(NDArray<Type> y, NDArray<Type> p)
        {
            var p0 = NumDN.Clamp(p, 1e-7, 1 - 1e-7);
            return -y * NumDN.Log(p0) - (1 - y) * NumDN.Log(1 - p0);
        }
    }

    public class SGD<Type> : IOptimizer<Type>
    {
        private readonly double lr = 0.01;
        private readonly double momentum;
        private NDArray<Type> weightsUpdate;

        public SGD() { }
        public SGD(double lr = 0.01, double momentum = 0.0)
        {
            this.lr = lr;
            this.momentum = momentum;
        }

        public IOptimizer<Type> Clone() => new SGD<Type>(lr, momentum);

        public NDArray<Type> Update(NDArray<Type> w, NDArray<Type> g)
        {
            if (weightsUpdate == null)
                weightsUpdate = NDArray<Type>.Zeros(w.Shape);

            if (Math.Abs(momentum) > 1e-15)
                weightsUpdate = momentum * w + (1 - momentum) * g;

            return w - lr * g;
        }
    }
}
