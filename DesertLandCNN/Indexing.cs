using System;
using System.Collections.Generic;
using System.Linq;

namespace DesertLandCNN
{
    public static class ExtensionIndexes
    {
        #region IndexInfo
        public enum IndexType { ndarray = 0, slice = 1 }

        public abstract class IndexInfo
        {
            public int idxB, idxE;
            public object content;
            public abstract IndexType indexType { get; }
            public abstract int GetContent(params int[] info);
        }

        public class IndexInfoSlice : IndexInfo
        {
            public override IndexType indexType => IndexType.slice;

            public override int GetContent(params int[] info)
            {
                int i = info[0];
                return (content as int[])[i];
            }

            public override string ToString()
            {
                var lt = content as int[];
                return $"[{indexType}({lt.Glue()}) Pos {idxB}=>{idxE}]";
            }
        }

        public class IndexInfoNDarray : IndexInfo
        {
            public override IndexType indexType => IndexType.ndarray;

            public override int GetContent(params int[] info)
            {
                var nD = content as NDArray<int>;
                int idx = nD.BcArray2Index(info);
                return nD.items[idx];
            }

            public override string ToString()
            {
                var nD = content as NDArray<int>;
                return $"[{indexType}({nD.Shape.Glue()}) Pos {idxB}=>{idxE}]";
            }
        }
        #endregion

        public static int[] GetSlice(int length, string s)
        {
            s = s.Trim();
            int nb = s.Count(a => a == ':');
            if (nb == 0 || nb > 2)
                throw new ArgumentException();

            if (nb == 1) s += ":1";

            string[] split = s.Split(new char[] { ':' }, StringSplitOptions.None).ToArray();
            if (string.IsNullOrEmpty(split[0])) split[0] = "0";
            if (string.IsNullOrEmpty(split[1])) split[1] = $"{length}";

            int start = int.Parse(split[0]);
            int end = Math.Min(length - 1, int.Parse(split[1]) - 1);
            int step = int.Parse(split[2]);

            int sz = (end + 1 - start) / step;
            return Enumerable.Range(0, sz).Select(i => start + i * step).ToArray();
        }

        public static (NDArray<Type>, List<IndexInfo>, int) ReshapeAndIndexInfos<Type>(this NDArray<Type> nD, params object[] args)
        {
            if (nD.Shape.Length + args.Count(i => i == NumDN.NewAxis) < args.Length)
                throw new ArgumentException("Too many indices for array");

            List<int> nshape = nD.Shape.ToList();
            List<object> args0 = args.ToList();
            for (int k = 0; k < args0.Count; ++k)
            {
                var v = args0[k];
                if (v == NumDN.NewAxis)
                {
                    if (k >= nshape.Count) nshape.Add(1);
                    else nshape.Insert(k, 1);
                    args0[k] = ":";
                }
                else if (v is int)
                    args0[k] = new NDArray<int>(new int[] { (int)v }, new int[] { 1 });
            }

            var nD0 = nD.ReShape(nshape.ToArray());
            if (nD0.Shape.Length > args.Length)
            {
                int diff = nD0.Shape.Length - args.Length;
                args0.AddRange(Enumerable.Repeat(":", diff));
            }

            int ndCount = 0;
            var indexInfos = new List<IndexInfo>(args0.Count);
            for (int k = 0; k < args0.Count; ++k)
            {
                var o = args0[k];
                if (o is string)
                {
                    IndexInfoSlice slice = new IndexInfoSlice();
                    slice.idxB = slice.idxE = k;
                    slice.content = GetSlice(nD0.Shape[k], o.ToString());
                    indexInfos.Add(slice);
                }
                else
                {
                    IndexInfoNDarray nDarray = new IndexInfoNDarray();
                    nDarray.idxB = nDarray.idxE = k;
                    nDarray.content = o;
                    indexInfos.Add(nDarray);
                    ++ndCount;
                }
            }

            return (nD0, indexInfos, ndCount);
        }

        public static int PosBcArray(List<IndexInfo> indexInfos)
        {
            int posBC = 0, prev = -1, curr = 0;
            for (int k = 0; k < indexInfos.Count; ++k)
            {
                var o = indexInfos[k];
                if (o.indexType == IndexType.slice) continue;
                if (prev == -1) posBC = k;

                curr = k;
                if (prev != -1 && curr != prev + 1)
                {
                    posBC = 0;
                    indexInfos.Sort((a, b) => a.indexType.CompareTo(b.indexType));
                    break;
                }
                prev = k;
            }

            return posBC;
        }

        public static (int[], int) FinalShapeAndBcLength(List<IndexInfo> indexInfos, int posBC, int ndCount)
        {
            if (ndCount == 0)
            {
                int[] nshape0 = indexInfos.Select(o => (o.content as int[]).Length).ToArray();
                return (nshape0, 0);
            }

            List<int> nshape = new List<int>();
            var arrBcShape = NumDN.BroadCasting(indexInfos.Skip(posBC).Take(ndCount).Select(a => a.content as NDArray<int>).ToArray());

            bool bcAdded = false;
            for (int k = 0, pos = 0; k < indexInfos.Count; ++k)
            {
                var o = indexInfos[k];
                if (o.indexType == IndexType.slice)
                {
                    nshape.Add((o.content as int[]).Length);
                    o.idxE = pos++;
                }
                else
                {
                    o.idxE = posBC;
                    if (!bcAdded)
                    {
                        nshape.AddRange(arrBcShape);
                        pos += arrBcShape.Length;
                        bcAdded = true;
                    }
                }
            }

            return (nshape.ToArray(), arrBcShape.Length);
        }

        public static NDArray<Type> GetIndexes<Type>(this NDArray<Type> nD, params object[] args)
        {
            (var ndB, var infos, var ndCount) = nD.ReshapeAndIndexInfos(args);
            int posBC = PosBcArray(infos);
            (var nshape, var bcLength) = FinalShapeAndBcLength(infos, posBC, ndCount);

            int[] tmpBcArr = new int[bcLength];
            var ndE = new NDArray<Type>(nshape);

            for (int idxE = 0; idxE < ndE.items.Length; ++idxE)
            {
                ndE.Index2Array(idxE);
                if (tmpBcArr.Length != 0)
                {
                    int pos = infos[posBC].idxE;
                    for (int k = 0; k < tmpBcArr.Length; ++k)
                        tmpBcArr[k] = ndE.tmpArr[pos + k];
                }

                foreach (var o in infos)
                {
                    if (o.indexType == IndexType.slice)
                        ndB.tmpArr[o.idxB] = o.GetContent(ndE.tmpArr[o.idxE]);
                    else
                        ndB.tmpArr[o.idxB] = o.GetContent(tmpBcArr);
                }

                int idxB = ndB.Array2Index();
                ndE.items[idxE] = ndB.items[idxB];
            }

            return ndE;
        }

        public static (NDArray<Type>, NDArray<Type>, Func<Type, Type, Type>) AddAt<Type>(NDArray<Type> nD0, NDArray<Type> nD1) => (nD0, nD1, NDArray<Type>.OpsT.Add);
        public static (NDArray<Type>, NDArray<Type>, Func<Type, Type, Type>) AddAt<Type>(NDArray<Type> nD0, Type v) => (nD0, new NDArray<Type>(v, new int[] { 1 }), NDArray<Type>.OpsT.Add);

        public static NDArray<Type> Indexes<Type>(this (NDArray<Type>, NDArray<Type>, Func<Type, Type, Type>) tuple, params object[] args)
        {
            var nD = tuple.Item1;
            var ndA = tuple.Item2;
            var ops = tuple.Item3;

            var ndF = new NDArray<Type>(nD);

            (var ndB, var infos, var ndCount) = nD.ReshapeAndIndexInfos(args);
            int posBC = PosBcArray(infos);
            (var nshape, var bcLength) = FinalShapeAndBcLength(infos, posBC, ndCount);

            int[] tmpBcArr = new int[bcLength];
            var ndE = new NDArray<Type>(nshape);
            NumDN.BroadCasting(ndA, ndE);

            for (int idxE = 0; idxE < ndE.items.Length; ++idxE)
            {
                ndE.Index2Array(idxE);
                if (tmpBcArr.Length != 0)
                {
                    int pos = infos[posBC].idxE;
                    for (int k = 0; k < tmpBcArr.Length; ++k)
                        tmpBcArr[k] = ndE.tmpArr[pos + k];
                }

                foreach (var o in infos)
                {
                    if (o.indexType == IndexType.slice)
                        ndB.tmpArr[o.idxB] = o.GetContent(ndE.tmpArr[o.idxE]);
                    else
                        ndB.tmpArr[o.idxB] = o.GetContent(tmpBcArr);
                }

                int idxB = ndB.Array2Index();
                int idxA = ndA.BcArray2Index(ndE.tmpArr);
                ndF.items[idxB] = ops(ndF.items[idxB], ndA.items[idxA]);
            }

            return ndF;
        }

    }
}
