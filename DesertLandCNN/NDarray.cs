using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DesertLandCNN
{
    public static class ExtOps
    {
        public static string Glue<T>(this IEnumerable<T> ts, string sep = " ", string format = "{0}", string globalformat = "{0}") => string.Format(globalformat, string.Join(sep, ts.Select(a => string.Format(format, a))));
    }

    public abstract class Ops<Type>
    {
        public Type Zero;
        public abstract Type Neg(Type a);
        public abstract Type Add(Type a, Type b);
        public abstract Type Add<U>(Type a, U b);
        public abstract Type Add<U>(U a, Type b);
        public abstract Type Sub(Type a, Type b);
        public abstract Type Sub<U>(Type a, U b);
        public abstract Type Sub<U>(U a, Type b);
        public abstract Type Mul(Type a, Type b);
        public abstract Type Mul<U>(Type a, U b);
        public abstract Type Mul<U>(U a, Type b);
        public abstract Type Div(Type a, Type b);
        public abstract Type Div<U>(Type a, U b);
        public abstract Type Div<U>(U a, Type b);

        public abstract Type Exp(Type x);
        public abstract Type Log(Type x);
        public abstract Type Abs(Type x);
        public abstract Type Sqrt(Type x);
        public abstract Type Sq(Type x);
        public abstract Type Tanh(Type x);
        public abstract Type Sigmoid(Type x);
        public abstract Type Clamp(Type x, double min, double max);
    }

    public class OpsInt : Ops<int>
    {
        public OpsInt() { Zero = 0; }
        public override int Neg(int a) => -a;
        public override int Add(int a, int b) => a + b;
        public override int Add<U>(int a, U b) => a + Convert.ToInt32(b);
        public override int Add<U>(U a, int b) => Convert.ToInt32(a) + b;
        public override int Sub(int a, int b) => a - b;
        public override int Sub<U>(int a, U b) => a - Convert.ToInt32(b);
        public override int Sub<U>(U a, int b) => Convert.ToInt32(a) - b;
        public override int Mul(int a, int b) => a * b;
        public override int Mul<U>(int a, U b) => a * Convert.ToInt32(b);
        public override int Mul<U>(U a, int b) => Convert.ToInt32(a) * b;
        public override int Div(int a, int b) => a / b;
        public override int Div<U>(int a, U b) => a / Convert.ToInt32(b);
        public override int Div<U>(U a, int b) => Convert.ToInt32(a) / b;

        public override int Exp(int x) => throw new NotImplementedException();
        public override int Log(int x) => throw new NotImplementedException();
        public override int Abs(int x) => Math.Abs(x);
        public override int Sqrt(int x) => throw new NotImplementedException();
        public override int Sq(int x) => x * x;
        public override int Tanh(int x) => throw new NotImplementedException();
        public override int Sigmoid(int x) => throw new NotImplementedException();
        public override int Clamp(int x, double min, double max) => (int)Math.Min(max, Math.Max(min, x));

    }

    public class OpsFloat : Ops<float>
    {
        public OpsFloat() { Zero = 0f; }
        public override float Neg(float a) => -a;
        public override float Add(float a, float b) => a + b;
        public override float Add<U>(float a, U b) => a + Convert.ToSingle(b);
        public override float Add<U>(U a, float b) => Convert.ToSingle(a) + b;
        public override float Sub(float a, float b) => a - b;
        public override float Sub<U>(float a, U b) => a - Convert.ToSingle(b);
        public override float Sub<U>(U a, float b) => Convert.ToSingle(a) - b;
        public override float Mul(float a, float b) => a * b;
        public override float Mul<U>(float a, U b) => a * Convert.ToSingle(b);
        public override float Mul<U>(U a, float b) => Convert.ToSingle(a) * b;
        public override float Div(float a, float b) => a / b;
        public override float Div<U>(float a, U b) => a / Convert.ToSingle(b);
        public override float Div<U>(U a, float b) => Convert.ToSingle(a) / b;

        public override float Exp(float x) => (float)Math.Exp(x);
        public override float Log(float x) => (float)Math.Log(x);
        public override float Abs(float x) => Math.Abs(x);
        public override float Sqrt(float x) => (float)Math.Sqrt(x);
        public override float Sq(float x) => x * x;
        public override float Tanh(float x) => (float)Math.Tanh(x);
        public override float Sigmoid(float x) => (float)(1 / (1 + Math.Exp(-x)));
        public override float Clamp(float x, double min, double max) => (float)Math.Min(max, Math.Max(min, x));
    }

    public class OpsDouble : Ops<double>
    {
        public OpsDouble() { Zero = 0.0; }
        public override double Neg(double a) => -a;
        public override double Add(double a, double b) => a + b;
        public override double Add<U>(double a, U b) => a + Convert.ToDouble(b);
        public override double Add<U>(U a, double b) => Convert.ToDouble(a) + b;
        public override double Sub(double a, double b) => a - b;
        public override double Sub<U>(double a, U b) => a - Convert.ToDouble(b);
        public override double Sub<U>(U a, double b) => Convert.ToDouble(a) - b;
        public override double Mul(double a, double b) => a * b;
        public override double Mul<U>(double a, U b) => a * Convert.ToDouble(b);
        public override double Mul<U>(U a, double b) => Convert.ToDouble(a) * b;
        public override double Div(double a, double b) => a / b;
        public override double Div<U>(double a, U b) => a / Convert.ToDouble(b);
        public override double Div<U>(U a, double b) => Convert.ToDouble(a) / b;

        public override double Exp(double x) => Math.Exp(x);
        public override double Log(double x) => Math.Log(x);
        public override double Abs(double x) => Math.Abs(x);
        public override double Sqrt(double x) => Math.Sqrt(x);
        public override double Sq(double x) => x * x;
        public override double Tanh(double x) => Math.Tanh(x);
        public override double Sigmoid(double x) => (1 / (1 + Math.Exp(-x)));
        public override double Clamp(double x, double min, double max) => Math.Min(max, Math.Max(min, x));
    }

    public class NDArray<Type>
    {

        #region Constructor
        public static Ops<Type> OpsT;

        static NDArray()
        {
            if (typeof(Type) == typeof(int))
                OpsT = new OpsInt() as Ops<Type>;
            else if (typeof(Type) == typeof(float))
                OpsT = new OpsFloat() as Ops<Type>;
            else if (typeof(Type) == typeof(double))
                OpsT = new OpsDouble() as Ops<Type>;
            else
                throw new ArgumentException($"{typeof(Type).Name} is not supported. Only int, float or double");
        }

        public Type[] items;
        public int[] Shape { get; private set; }
        public int[] tmpArr { get; private set; }

        public NDArray(params int[] shape)
        {
            int dim = NumDN.ShapeLength(shape);
            if (dim <= 0)
                throw new ArgumentException();

            Shape = shape.ToArray();
            tmpArr = new int[Shape.Length];
            items = new Type[dim];
        }

        public NDArray(Type v, int[] shape)
        {
            int dim = NumDN.ShapeLength(shape);
            if (dim <= 0)
                throw new ArgumentException();

            Shape = shape.ToArray();
            tmpArr = new int[Shape.Length];
            items = Enumerable.Repeat(v, dim).ToArray();
        }

        public NDArray(Type[] nd, int[] shape)
        {
            int dim = NumDN.ShapeLength(shape);
            if (dim <= 0)
                throw new ArgumentException();

            Shape = shape.ToArray();
            tmpArr = new int[Shape.Length];
            items = nd.ToArray();
        }

        public NDArray(Type[,] nD)
        {
            if (nD == null)
                throw new ArgumentException();

            int dim0 = nD.GetLength(0);
            int dim1 = nD.GetLength(1);
            int dim = dim0 * dim1;
            Shape = new int[] { dim0, dim1 };
            tmpArr = new int[Shape.Length];
            items = new Type[dim];

            int a = 0;
            for (int i = 0; i < dim0; ++i)
                for (int j = 0; j < dim1; ++j)
                    items[a++] = nD[i, j];
        }

        public NDArray(NDArray<Type> nD)
        {
            if (nD == null)
                throw new ArgumentException();

            Shape = nD.Shape.ToArray();
            tmpArr = new int[Shape.Length];
            items = nD.items.ToArray();
        }

        public NDArray<V> Apply<V>(Func<Type, V> func)
        {
            var nd = new NDArray<V>(Shape);
            nd.items = items.Select(func).ToArray();
            return nd;
        }

        public NDArray<Type> this[int i]
        {
            get
            {
                if (i >= Shape[0] || i < -1)
                    throw new ArgumentException();

                var dim0 = NumDN.ShapeLength(Shape);
                var dim1 = dim0 / Shape[0];
                var nshape = Shape.Skip(1).ToArray();

                var start = i * dim1;
                var nd = new NDArray<Type>(nshape);
                for (int j = 0, k = start; j < dim1; ++j, ++k)
                    nd.items[j] = items[k];

                return nd;
            }
            set
            {
                var dim0 = NumDN.ShapeLength(Shape);
                var dim1 = NumDN.ShapeLength(value.Shape);
                if (dim0 / dim1 != Shape[0] || i >= Shape[0] || i < 0)
                    throw new ArgumentException();

                var start = i * dim1;
                for (int j = 0, k = start; j < dim1; ++j, ++k)
                    items[k] = value.items[j];
            }
        }

        public NDArray<V> Cast<V>() => Apply(i => (V)Convert.ChangeType(i, typeof(V)));

        #endregion


        #region Display
        private string PrettyDisplay(string fmt, int depth = 0)
        {
            if (Shape.Length == 1)
                return $"[{items.Glue(" ", fmt)}]";

            StringBuilder sb = new StringBuilder();
            string space = Enumerable.Repeat("", depth + 2).Glue();

            for (int k = 0; k < Shape[0]; ++k)
            {
                string b = k == 0 ? "[" : space;
                string e = k == Shape[0] - 1 ? "]" : space;

                if (k != Shape[0] - 1)
                    sb.AppendLine(b + this[k].PrettyDisplay(fmt, depth + 1) + e);
                else
                    sb.Append(b + this[k].PrettyDisplay(fmt, depth + 1) + e);
            }

            return sb.ToString();
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            if (NumDN.DebugNumPy)
            {
                string dbg = $" : np.array([{items.Glue(",")}]).reshape({Shape.Glue(",")})";
                sb.AppendLine($"NDArray<{typeof(Type).Name}> {Shape.Glue("x")}{dbg}");
            }

            int mx = items.Select(i => i.ToString()).Max(i => i.Length);
            if (mx > 5)
                mx = items.Select(i => $"{i:F6}").Max(i => i.Length);

            string fmt = mx < 5 ? $"{{0, {mx}}}" : $"{{0, {mx}:F6}}";
            sb.Append(PrettyDisplay(fmt));
            if (NumDN.DebugNumPy)
                sb.AppendLine();

            return sb.ToString();
        }
        #endregion

        #region Transpose
        internal int Array2Index()
        {
            int index = 0;
            for (int k = 0; k < Shape.Length; ++k)
                index = index * Shape[k] + tmpArr[k];

            return index;
        }

        internal void Index2Array(int index)
        {
            int index0 = index;
            for (int k = Shape.Length - 1; k >= 0; --k)
            {
                tmpArr[k] = index0 % Shape[k];
                index0 /= Shape[k];
            }
        }

        public NDArray<Type> transpose(params int[] axis)
        {
            if (Enumerable.Range(0, Shape.Length).Any(i => !axis.Contains(i)))
                throw new ArgumentException();

            var nshape = axis.Select(i => Shape[i]).ToArray();
            var nD = new NDArray<Type>(nshape);

            var sLength = Shape.Length;

            for (int i = 0; i < items.Length; ++i)
            {
                Index2Array(i);
                for (int k = 0; k < sLength; ++k)
                    nD.tmpArr[k] = tmpArr[axis[k]];

                int j = nD.Array2Index();
                nD.items[j] = items[i];
            }

            return nD;
        }

        public NDArray<Type> T
        {
            get
            {
                //var dim = NumDN.ShapeLength(Shape);
                //var nshape = Shape.Reverse().ToArray();
                //var nd = new NDArray<Type>(nshape);

                //var sLength = Shape.Length;
                //var idxArr = new int[sLength];
                //var idxTArr = new int[sLength];

                //for (int i = 0; i < dim; ++i)
                //{
                //    Index2Array(i, idxArr);
                //    for (int k = 0; k < sLength; ++k)
                //        idxTArr[sLength - 1 - k] = idxArr[k];

                //    nd.Array2Index(idxTArr, out int j);
                //    nd.items[j] = items[i];
                //}

                //return nd;
                var axis = Enumerable.Range(0, Shape.Length).Reverse().ToArray();
                return transpose(axis);
            }
        }
        #endregion

        #region ReShape
        public NDArray<Type> ReShape(params int[] args)
        {
            int oldDim = NumDN.ShapeLength(Shape);
            int newDim = NumDN.ShapeLength(args);
            if (oldDim != newDim)
                throw new ArgumentException();

            var ndarr = new NDArray<Type>(this);
            ndarr.Shape = args.Length != 0 ? args.ToArray() : new int[] { 1 };
            ndarr.tmpArr = new int[ndarr.Shape.Length];
            return ndarr;
        }

        public void ReShapeInplace(params int[] args)
        {
            int oldDim = NumDN.ShapeLength(Shape);
            int newDim = NumDN.ShapeLength(args);
            if (oldDim != newDim)
                throw new ArgumentException();

            Shape = args.Length != 0 ? args.ToArray() : new int[] { 1 };
            tmpArr = new int[Shape.Length];
        }
        #endregion

        #region Broadcast, ElementOps, TensorDot
        internal void BcIndex2Array(int index, int[] nshape, int[] outArray)
        {
            int index0 = index;
            for (int i = Shape.Length - 1, j = nshape.Length - 1, k = outArray.Length - 1; i >= 0 && j >= 0 && k >= 0; --i, --j, --k)
            {
                outArray[k] = (index0 % Shape[i]) % nshape[j];
                index0 /= Shape[i];
            }
        }

        internal int BcArray2Index(int[] arrIdx)
        {
            for (int k = Shape.Length - 1, k0 = arrIdx.Length - 1; k >= 0; --k, --k0)
                tmpArr[k] = arrIdx[k0] % Shape[k];

            //Console.WriteLine($"#      tmpArray:({tmpArray.Glue()})");
            return Array2Index();
        }

        public NDArray<Type> Broadcast(params int[] args)
        {
            int sLength = Shape.Length;
            int nLength = args.Length;
            int mLength = Math.Max(sLength, nLength);

            int[] nshape = new int[mLength];
            for (int k = mLength - 1, i = sLength - 1, j = nLength - 1; k >= 0; --k, --i, --j)
            {
                int idx0 = i < 0 ? 1 : Shape[i];
                int idx1 = j < 0 ? 1 : args[j];
                if (idx0 != idx1 && idx0 != 1 && idx1 != 1)
                    throw new ArgumentException($"Cannot broadcast {Shape.Glue("x")} with {args.Glue("x")}");

                nshape[k] = Math.Max(idx0, idx1);
            }

            var nd = new NDArray<Type>(nshape);
            for (int k = 0; k < nd.items.Length; ++k)
            {
                nd.BcIndex2Array(k, Shape, tmpArr);
                int j = Array2Index();
                nd.items[k] = items[j];
            }

            return nd;
        }

        static NDArray<Type> ElementOps(NDArray<Type> nD0, NDArray<Type> nD1, Func<Type, Type, Type> func)
        {
            int sLength0 = nD0.Shape.Length;
            int sLength1 = nD1.Shape.Length;
            int mLength = Math.Max(sLength0, sLength1);

            int[] nshape = new int[mLength];
            for (int k = mLength - 1, i = sLength0 - 1, j = sLength1 - 1; k >= 0; --k, --i, --j)
            {
                int idx0 = i < 0 ? 1 : nD0.Shape[i];
                int idx1 = j < 0 ? 1 : nD1.Shape[j];
                if (idx0 != idx1 && idx0 != 1 && idx1 != 1)
                    throw new ArgumentException($"Cannot broadcast {nD0.Shape.Glue("x")} with {nD1.Shape.Glue("x")}");

                nshape[k] = Math.Max(idx0, idx1);
            }

            var nd = new NDArray<Type>(nshape);

            for (int k = 0; k < nd.items.Length; ++k)
            {
                nd.BcIndex2Array(k, nD0.Shape, nD0.tmpArr);
                nd.BcIndex2Array(k, nD1.Shape, nD1.tmpArr);
                int i0 = nD0.Array2Index();
                int i1 = nD1.Array2Index();
                nd.items[k] = func(nD0.items[i0], nD1.items[i1]);
            }

            return nd;
        }

        public static NDArray<Type> Dot(NDArray<Type> nD0, NDArray<Type> nD1)
        {
            if (nD0.Shape.Length == 1)
                nD0 = nD0.ReShape(1, nD0.Shape[0]);

            if (nD1.Shape.Length == 1)
                nD1 = nD1.ReShape(nD1.Shape[0], 1);

            int length0 = nD0.Shape.Length;
            int length1 = nD1.Shape.Length;
            int piv = nD0.Shape.Last();

            if (piv != nD1.Shape[length1 - 2])
                throw new ArgumentException($"Cannot multiply {nD0.Shape.Glue("x")} and {nD1.Shape.Glue("x")}");

            int[] nshape = new int[length0 + length1 - 2];
            int[] idxInfos = new int[length0 + length1 - 2];
            for (int k = 0, k0 = 0; k < length0 + length1; ++k)
            {
                if (k == length0 - 1 || k == length0 + length1 - 2) continue;
                if (k < length0 - 1) nshape[k] = nD0.Shape[idxInfos[k] = k];
                else nshape[k0] = nD1.Shape[idxInfos[k0] = k - length0];
                ++k0;
            }

            int dim = NumDN.ShapeLength(nshape);
            var nd = new NDArray<Type>(nshape);

            for (int m = 0; m < nd.items.Length; ++m)
            {
                Type sum = OpsT.Zero;
                nd.Index2Array(m);

                for (int k = 0; k < nshape.Length; ++k)
                {
                    if (k < length0 - 1) nD0.tmpArr[idxInfos[k]] = nd.tmpArr[k];
                    else nD1.tmpArr[idxInfos[k]] = nd.tmpArr[k];
                }

                for (int i = 0; i < piv; ++i)
                {
                    nD0.tmpArr[length0 - 1] = nD1.tmpArr[length1 - 2] = i;
                    int idx0 = nD0.Array2Index();
                    int idx1 = nD1.Array2Index();

                    sum = OpsT.Add(sum, OpsT.Mul(nD0.items[idx0], nD1.items[idx1]));
                }

                nd.items[m] = sum;
            }

            return nd;
        }
        #endregion

        #region Static Operations with Constants

        public static NDArray<Type> Zeros(params int[] shape) => new NDArray<Type>(shape);
        public static NDArray<Type> Ones(params int[] shape) => new NDArray<Type>((Type)Convert.ChangeType(1, typeof(Type)), shape);

        public static NDArray<Type> operator +(double v, NDArray<Type> nD) => nD.Apply(i => OpsT.Add(v, i));
        public static NDArray<Type> operator +(NDArray<Type> nD, double v) => nD.Apply(i => OpsT.Add(i, v));
        public static NDArray<Type> operator -(NDArray<Type> nD) => nD.Apply(OpsT.Neg);
        public static NDArray<Type> operator -(double v, NDArray<Type> nD) => nD.Apply(i => OpsT.Sub(v, i));
        public static NDArray<Type> operator -(NDArray<Type> nD, double v) => nD.Apply(i => OpsT.Sub(i, v));
        public static NDArray<Type> operator *(double v, NDArray<Type> nD) => nD.Apply(i => OpsT.Mul(v, i));
        public static NDArray<Type> operator *(NDArray<Type> nD, double v) => nD.Apply(i => OpsT.Mul(i, v));
        public static NDArray<Type> operator /(double v, NDArray<Type> nD) => nD.Apply(i => OpsT.Div(v, i));
        public static NDArray<Type> operator /(NDArray<Type> nD, double v) => nD.Apply(i => OpsT.Div(i, v));

        public static NDArray<Type> operator +(NDArray<Type> a, NDArray<Type> b) => ElementOps(a, b, (ia, ib) => OpsT.Add(ia, ib));
        public static NDArray<Type> operator -(NDArray<Type> a, NDArray<Type> b) => ElementOps(a, b, (ia, ib) => OpsT.Sub(ia, ib));
        public static NDArray<Type> operator *(NDArray<Type> a, NDArray<Type> b) => ElementOps(a, b, (ia, ib) => OpsT.Mul(ia, ib));
        public static NDArray<Type> operator /(NDArray<Type> a, NDArray<Type> b) => ElementOps(a, b, (ia, ib) => OpsT.Div(ia, ib));

        #endregion

    }

    public static class NumDN
    {
        public static bool DebugNumPy = false;

        public const object NewAxis = null;

        #region Static Utilities
        private static Random random;
        private static Random GetRandom
        {
            get
            {
                if (random == null)
                    //random = new Random((int)DateTime.Now.Ticks);
                    random = new Random(123);

                return random;
            }
        }

        public static int ShapeLength(int[] shape) => shape.Aggregate(1, (a, i) => a * i);

        public static NDArray<V> Apply<U, V>(NDArray<U> a, NDArray<U> b, Func<U, U, V> func)
        {
            int dim = ShapeLength(a.Shape);
            if (dim != ShapeLength(b.Shape))
                throw new ArgumentException();

            var nd = new NDArray<V>(a.Shape);
            nd.items = Enumerable.Range(0, dim).Select(i => func(a.items[i], b.items[i])).ToArray();
            return nd;
        }

        #endregion

        #region Random Sample
        public static NDArray<int> UniformInt(int min, int max, params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[1] { 1 };

            int dim = ShapeLength(shape);
            if (dim <= 0 || min >= max)
                throw new ArgumentException();

            int[] r0 = Enumerable.Range(0, dim).Select(i => GetRandom.Next(min, max)).ToArray();
            NDArray<int> rs = new NDArray<int>(r0, shape);

            return rs;
        }

        public static NDArray<float> UniformFloat(float min, float max, params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            int dim = ShapeLength(shape);
            if (dim <= 0 || min >= max)
                throw new ArgumentException();

            float[] r0 = Enumerable.Range(0, dim).Select(i => min + (max - min) * (float)GetRandom.NextDouble()).ToArray();
            NDArray<float> rs = new NDArray<float>(r0, shape);

            return rs;
        }

        public static NDArray<double> UniformDouble(double min, double max, params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            int dim = ShapeLength(shape);
            if (dim <= 0 || min >= max)
                throw new ArgumentException();

            double[] r0 = Enumerable.Range(0, dim).Select(i => min + (max - min) * GetRandom.NextDouble()).ToArray();
            NDArray<double> rs = new NDArray<double>(r0, shape);

            return rs;
        }

        public static NDArray<Type> Uniform<Type>(double min, double max, params int[] shape)
        {
            if (shape.Length == 0)
                shape = new int[] { 1 };

            int dim = ShapeLength(shape);
            if (dim <= 0 || min >= max)
                throw new ArgumentException();

            double[] r0 = Enumerable.Range(0, dim).Select(i => min + (max - min) * GetRandom.NextDouble()).ToArray();
            NDArray<double> rs = new NDArray<double>(r0, shape);
            var rt = rs.Cast<Type>();

            return rt;
        }

        #endregion

        public static NDArray<Type> Exp<Type>(NDArray<Type> nD) => nD.Apply(NDArray<Type>.OpsT.Exp);
        public static NDArray<Type> Log<Type>(NDArray<Type> nD) => nD.Apply(NDArray<Type>.OpsT.Log);
        public static NDArray<Type> Abs<Type>(NDArray<Type> nD) => nD.Apply(NDArray<Type>.OpsT.Abs);
        public static NDArray<Type> Sqrt<Type>(NDArray<Type> nD) => nD.Apply(NDArray<Type>.OpsT.Sqrt);
        public static NDArray<Type> Sq<Type>(NDArray<Type> nD) => nD.Apply(NDArray<Type>.OpsT.Sq);
        public static NDArray<Type> Tanh<Type>(NDArray<Type> nD) => nD.Apply(NDArray<Type>.OpsT.Tanh);
        public static NDArray<Type> Sigmoid<Type>(NDArray<Type> nD) => nD.Apply(NDArray<Type>.OpsT.Sigmoid);
        public static NDArray<Type> Clamp<Type>(NDArray<Type> nD, double min, double max) => nD.Apply(x => NDArray<Type>.OpsT.Clamp(x, min, max));

        public static double Max<Type>(NDArray<Type> nD) => nD.items.Max(i => Convert.ToDouble(i));
        public static double SumDouble<Type>(NDArray<Type> nD) => nD.items.Sum(i => Convert.ToDouble(i));
        public static double Mean<Type>(NDArray<Type> nD) => nD.items.Average(i => Convert.ToDouble(i));

        public static NDArray<Type> Sum<Type>(NDArray<Type> nD, params int[] axis) => Sum(nD, true, axis);

        public static NDArray<Type> Sum<Type>(NDArray<Type> nD, bool keepDims, params int[] axis)
        {
            if (axis.Length == 0)
                axis = Enumerable.Range(0, nD.Shape.Length).ToArray();

            var nshape0 = nD.Shape.ToArray();
            foreach (var a in axis)
                nshape0[a] = 1;

            var nD0 = new NDArray<Type>(nshape0);
            int[] arrIdx0 = new int[nD.Shape.Length];
            for (int k = 0; k < nD.items.Length; ++k)
            {
                nD.BcIndex2Array(k, nshape0, nD0.tmpArr);
                int idx = nD0.Array2Index();
                nD0.items[idx] = NDArray<Type>.OpsT.Add(nD.items[k], nD0.items[idx]);
            }

            if (!keepDims)
            {
                foreach (var a in axis)
                    nshape0[a] = -1;

                nshape0 = nshape0.Where(a => a != -1).ToArray();
                nD0 = nD0.ReShape(nshape0);
            }

            return nD0;
        }

        public static NDArray<Type> Pad<Type>(NDArray<Type> nD, params (int, int)[] padding)
        {
            if (nD.Shape.Length != padding.Length)
                throw new ArgumentException();

            if (DebugNumPy)
            {
                Console.WriteLine("np.pad(a, ({0}), mode='constant')", string.Join(", ", padding));
            }

            var nshape = Enumerable.Range(0, nD.Shape.Length).Select(i => nD.Shape[i] + padding[i].Item1 + padding[i].Item2).ToArray();
            var nD0 = new NDArray<Type>(nshape);

            for (int idx = 0; idx < nD.items.Length; ++idx)
            {
                nD.Index2Array(idx);
                for (int i = 0; i < nD.Shape.Length; ++i)
                    nD0.tmpArr[i] = nD.tmpArr[i] + padding[i].Item1;

                int idx0 = nD0.Array2Index();
                nD0.items[idx0] = nD.items[idx];
            }

            return nD0;
        }

        public static int[] ARange(int k) => Enumerable.Range(0, k).ToArray();
        public static int[] Repeat(int i, int k) => Enumerable.Repeat(i, k).ToArray();
        public static int[] Repeat(int[] i, int k) => i.SelectMany(a => Repeat(a, k)).ToArray();
        public static int[] Tile(int[] i, int k) => Enumerable.Range(0, k).SelectMany(a => i).ToArray();

        static int FuncAxisFromLast<Type>(NDArray<Type> nD, int idx) => nD.Shape.Length - idx <= 0 ? 1 : nD.Shape[nD.Shape.Length - idx - 1];

        public static int[] BroadCasting<Type>(params NDArray<Type>[] ns)
        {
            if (ns.Length == 0)
                throw new ArgumentException();

            //if (DebugNumPy)
            //{
            //    Console.WriteLine("Broadcasting");
            //    foreach (var n in ns)
            //        Console.WriteLine(n);

            //    Console.Write("Result ");
            //}

            int[] nshape = new int[ns.Max(nd => nd.Shape.Length)];
            int sz = nshape.Length;
            for (int k = 0; k < sz; ++k)
            {
                int max = 1;
                for(int i = 0; i < ns.Length; ++i)
                {
                    int v = FuncAxisFromLast(ns[i], k);
                    if (v == 1 || v == max)
                        continue;

                    if (max == 1)
                        max = v;
                    else
                        throw new ArgumentException($"Cannot do BroadCasting {ns.Select(n => n.Shape.Glue(globalformat: "Shape({0})")).Glue(" and ")}");
                }
                nshape[sz - k - 1] = max;
            }

            return nshape;
        }
    }
}
