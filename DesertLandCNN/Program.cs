using System;
using System.Collections.Generic;
using System.Linq;

namespace DesertLandCNN
{
    class MainClass
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
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
            //a.GetAtIndexes(":");
            //a.GetAtIndexes(":2", "1:");
            //a.GetAtIndexes(":2", NumDN.NewAxis, "1:");
            //a.GetAtIndexes(NumDN.NewAxis, ":2", "1:");
            //a.GetAtIndexes(":2", "1:", NumDN.NewAxis);
            //a.GetAtIndexes(NumDN.NewAxis, ":2", "1:", NumDN.NewAxis);
            //a.GetAtIndexes(NumDN.NewAxis, ":2", "1:", NumDN.NewAxis, NumDN.NewAxis);
            //a.GetAtIndexes(NumDN.NewAxis, NumDN.NewAxis, ":2", "1:", NumDN.NewAxis);
            //a.GetAtIndexes(NumDN.NewAxis, ":2", NumDN.NewAxis, "1:", NumDN.NewAxis);
            //a.GetAtIndexes(NumDN.NewAxis, ":2", NumDN.NewAxis, NumDN.NewAxis, "1:", NumDN.NewAxis);
            //a.GetAtIndexes(NumDN.NewAxis, ":2", NumDN.NewAxis, NumDN.NewAxis);

            //var b = NumDN.UniformInt(0, 3, 2, 4, 3);
            //var c = NumDN.UniformInt(0, 5, 5, 4);
            ////var d = NumDN.UniformInt(0, 4, 2, 4);
            //Console.WriteLine(a);
            ////Console.WriteLine(b);
            //Console.WriteLine(c);
            //Console.WriteLine(a + c);
            //Console.WriteLine(NDArray<int>.Dot(a, c.T));
            ////Console.WriteLine(d);

            var x = NumDN.UniformInt(0, 10, 4, 5, 3, 6);
            var y = NumDN.UniformInt(0, 4, 3, 2);
            var z = NumDN.UniformInt(0, 5, 3, 2);
            Console.WriteLine(x);
            Console.WriteLine(y);
            Console.WriteLine(z);
            //x.GetAtIndexes(":", ":2", NumDN.NewAxis, ":", "3:");
            //x.GetAtIndexes(":", z);
            //x.GetAtIndexes(y, "1:3");
            x.GetAtIndexes(y, z);
            //x.GetAtIndexes(":", z, ":1", z);
            //x.GetAtIndexes(":", z, z, ":1");
            //x.GetAtIndexes(":", ":1", z, z);
            //x.GetAtIndexes(y, NumDN.NewAxis, z);
            //x.GetAtIndexes(NumDN.NewAxis, y, z);


            //Console.WriteLine(a.T);
            //Console.WriteLine(NDArray<int>.Dot(a, b));

            //a.GetAtIndexes(":", c);
            //a.GetAtIndexes(b, "1:3");
            //a.GetAtIndexes(b, c, "2:3");
            //a.GetAtIndexes(c, ":", c);
            //a.GetAtIndexes(":", c, c);
            //a.GetAtIndexes(c, c, c);
        }
    }
}
