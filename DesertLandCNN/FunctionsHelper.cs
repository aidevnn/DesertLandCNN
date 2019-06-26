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
        public NDArray<Type> Function(NDArray<Type> x) => x.Apply(x0 => NDArray<Type>.OpsT.GtE(x0, NDArray<Type>.OpsT.Zero) ? x0 : NDArray<Type>.OpsT.Zero);

        public NDArray<Type> Gradient(NDArray<Type> x) => x.Apply(x0 => NDArray<Type>.OpsT.GtE(x0, NDArray<Type>.OpsT.Zero) ? NDArray<Type>.OpsT.One : NDArray<Type>.OpsT.Zero);
    }

    public class ReLUActivation<Type> : IActivation<Type>
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
            int n = y.Shape[0];
            double s = 0;
            var v = NumDN.Abs(y - p).Apply(x => Convert.ToDouble(x) < 0.5 ? 1 : 0);
            for (int k = 0; k < n; ++k)
                s += Enumerable.Range(k * 10, 10).Select(i => v.items[i]).Sum() == 10 ? 1.0 : 0.0;

            //var m = (y - p).items.Select(x => Math.Abs(Convert.ToDouble(x)) < 0.7 ? 1.0 : 0.0).Average();
            return s;
        }

        public NDArray<Type> Grad(NDArray<Type> y, NDArray<Type> p)
        {
            var p0 = NumDN.Clamp(p, 1e-15, 1 - 1e-15);
            return -y / p0 + (1 - y) / (1 - p0);
        }

        public NDArray<Type> Loss(NDArray<Type> y, NDArray<Type> p)
        {
            var p0 = NumDN.Clamp(p, 1e-15, 1 - 1e-15);
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

    /*

    class Adam():
        def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
            self.learning_rate = learning_rate
            self.eps = 1e-8
            self.m = None
            self.v = None
            # Decay rates
            self.b1 = b1
            self.b2 = b2

        def update(self, w, grad_wrt_w):
            # If not initialized
            if self.m is None:
                self.m = np.zeros(np.shape(grad_wrt_w))
                self.v = np.zeros(np.shape(grad_wrt_w))
            
            self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
            self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

            m_hat = self.m / (1 - self.b1)
            v_hat = self.v / (1 - self.b2)

            self.w_updt = self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

            return w - self.w_updt
     
     */
    public class Adam<Type> : IOptimizer<Type>
    {
        readonly double learning_rate, b1, b2;
        const double eps = 1e-8;

        NDArray<Type> weightsUpdate, m, v;

        public Adam(double learning_rate = 0.001, double b1 = 0.9, double b2 = 0.999)
        {
            this.learning_rate = learning_rate;
            this.b1 = b1;
            this.b2 = b2;
        }

        public IOptimizer<Type> Clone() => new Adam<Type>(learning_rate, b1, b2);

        public NDArray<Type> Update(NDArray<Type> w, NDArray<Type> g)
        {
            if (m == null)
            {
                m = NDArray<Type>.Zeros(g.Shape);
                v = NDArray<Type>.Zeros(g.Shape);
            }

            m = b1 * m + (1 - b1) * g;
            v = b2 * v + (1 - b2) * NumDN.Sq(g);

            var mh = m / (1.0 - b1);
            var vh = v / (1.0 - b2);

            weightsUpdate = learning_rate * mh / (NumDN.Sqrt(vh) + eps);

            return w - weightsUpdate;
        }
    }
}
