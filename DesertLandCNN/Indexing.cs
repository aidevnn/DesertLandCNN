using System;
using System.Collections.Generic;
using System.Linq;

namespace DesertLandCNN
{
    public static class ExtensionIndexes
    {
        #region IndexInfo
        public enum IndexType { slice, ndarray }

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

        public class IndexManager<Type>
        {
            public NDArray<Type> Begin, End;
            List<IndexInfo> indexInfos;
            List<IndexInfoNDarray> infoNDarrays;
            int[] arrBcShape, tmpBcArr;

            public IndexManager(NDArray<Type> nD, object[] args)
            {
                Begin = new NDArray<Type>(nD);
                var args0 = CompleteArgsAndReshape(Begin, args);

                indexInfos = new List<IndexInfo>(args0.Length);
                infoNDarrays = new List<IndexInfoNDarray>(args0.Length);
                for (int k = 0; k < args0.Length; ++k)
                {
                    var o = args0[k];
                    if (o is string)
                    {
                        IndexInfoSlice slice = new IndexInfoSlice();
                        slice.idxB = slice.idxE = k;
                        slice.content = GetSlice(Begin.Shape[k], o.ToString());
                        indexInfos.Add(slice);
                    }
                    else
                    {
                        IndexInfoNDarray nDarray = new IndexInfoNDarray();
                        nDarray.idxB = nDarray.idxE = k;
                        nDarray.content = o;
                        infoNDarrays.Add(nDarray);
                        indexInfos.Add(nDarray);
                    }
                }
            }

            int PosBcArray()
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
                        break;
                    }
                    prev = k;
                }

                return posBC;
            }

            void PrepareEndArray()
            {
                tmpBcArr = new int[0];
                if (infoNDarrays.Count == 0)
                {
                    int[] nshape0 = indexInfos.Select(o => (o.content as int[]).Length).ToArray();
                    End = new NDArray<Type>(nshape0);
                    return;
                }

                int posBC = PosBcArray();
                List<int> nshape = new List<int>();
                arrBcShape = NumDN.BroadCasting(infoNDarrays.Select(a => a.content as NDArray<int>).ToArray());
                if (posBC == 0)
                    indexInfos = indexInfos.OrderByDescending(o => o.indexType == IndexType.ndarray).ToList();

                tmpBcArr = new int[arrBcShape.Length];
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

                End = new NDArray<Type>(nshape.ToArray());
            }

            public void Complete()
            {
                PrepareEndArray();
                Console.WriteLine($"Begin  Shape ({Begin.Shape.Glue()})");
                Console.WriteLine($"Finale Shape ({End.Shape.Glue()})");
                Console.WriteLine("Infos {0}", indexInfos.Glue("; "));
                Console.WriteLine();

                for(int idxE = 0; idxE < End.items.Length; ++idxE)
                {
                    End.Index2Array(idxE);
                    if (tmpBcArr.Length != 0)
                    {
                        int pos = infoNDarrays[0].idxE;
                        for(int k = 0; k < tmpBcArr.Length; ++k)
                            tmpBcArr[k] = End.tmpArr[pos + k];
                    }

                    foreach(var o in indexInfos)
                    {
                        if (o.indexType == IndexType.slice)
                            Begin.tmpArr[o.idxB] = o.GetContent(End.tmpArr[o.idxE]);
                        else
                            Begin.tmpArr[o.idxB] = o.GetContent(tmpBcArr);
                    }

                    int idxB = Begin.Array2Index();
                    End.items[idxE] = Begin.items[idxB];
                }
            }
        }
        #endregion

        #region IndexTable
        public abstract class IndexTable<Type>
        {
            public int[] EndShape, BeginShape;
            public NDArray<Type> Begin, End;

            public abstract IndexType indexType { get; }
            public abstract void Complete();
        }

        public class IndexTableNDarray<Type> : IndexTable<Type>
        {
            readonly object[] indexes;
            readonly int[] arrBcIdx, revIdxBc;
            readonly int bcIdx, bcLength;
            readonly Dictionary<int, int> revIdxSlice;

            object Filter(object arg, int k)
            {
                if (arg is string)
                    return GetSlice(Begin.Shape[k], arg.ToString());

                return arg;
            }

            public IndexTableNDarray(NDArray<Type> nD, object[] args)
            {
                revIdxSlice = new Dictionary<int, int>();
                Begin = new NDArray<Type>(nD);
                var args0 = CompleteArgsAndReshape(Begin, args);
                indexes = new object[args0.Length];

                IndexType it = IndexType.slice;
                int swap = 0;
                int bi = -1;
                List<NDArray<int>> nDs = new List<NDArray<int>>();
                List<int> revIdxBcL = new List<int>();
                for (int k = 0; k < args0.Length; ++k)
                {
                    var o = args0[k];
                    indexes[k] = Filter(o, k);
                    if (o is string)
                    {
                        if (it != IndexType.slice)
                        {
                            it = IndexType.slice;
                            ++swap;
                        }
                    }
                    else
                    {
                        nDs.Add(o as NDArray<int>);
                        revIdxBcL.Add(k);
                        if (bi == -1) bi = k;
                        if (it != IndexType.ndarray)
                        {
                            it = IndexType.ndarray;
                            ++swap;
                        }
                    }
                }

                var bcShape = NumDN.BroadCasting(nDs.ToArray());
                bcLength = bcShape.Length;
                arrBcIdx = new int[bcLength];
                revIdxBc = revIdxBcL.ToArray();
                List<int> nshapel = new List<int>();
                if (swap <= 2)
                {
                    bool bcAdded = false;
                    int pos = 0;
                    for (int k = 0; k < args0.Length; ++k)
                    {
                        var o = args0[k];
                        if (o is string)
                        {
                            nshapel.Add((indexes[k] as int[]).Length);
                            revIdxSlice[pos++] = k;
                        }
                        else if (!bcAdded)
                        {
                            bcIdx = k;
                            nshapel.AddRange(bcShape);
                            pos += bcLength;
                            bcAdded = true;
                        }
                    }
                }
                else
                {
                    bcIdx = 0;
                    nshapel.AddRange(bcShape);
                    int pos = bcLength;
                    for (int k = 0; k < args0.Length; ++k)
                    {
                        var o = args0[k];
                        if (o is string)
                        {
                            nshapel.Add((indexes[k] as int[]).Length);
                            revIdxSlice[pos++] = k;
                        }
                    }
                }

                End = new NDArray<Type>(nshapel.ToArray());

                Console.WriteLine(End.Shape.Glue(globalformat: "Final Shape ({0})"));
                Console.WriteLine(revIdxBc.Glue(globalformat: "BC Idx {0}"));
                Console.WriteLine(revIdxSlice.Select(e => $"({e.Key}=>{e.Value}, {args0[e.Value]})").Glue(globalformat: "Slice Idx {0}"));
            }

            public override IndexType indexType => IndexType.ndarray;

            void BcIndexes()
            {
                for(int k = 0; k < arrBcIdx.Length; ++k)
                    arrBcIdx[k] = End.tmpArr[bcIdx + k];

                for (int k = 0; k < revIdxBc.Length; ++k)
                {
                    var nD0 = indexes[revIdxBc[k]] as NDArray<int>;
                    int idx = nD0.BcArray2Index(arrBcIdx);
                    Begin.tmpArr[revIdxBc[k]] = nD0.items[idx];
                }
            }

            void SliceIndexes()
            {
                foreach(var e in revIdxSlice)
                {
                    var arr = indexes[e.Value] as int[];
                    Begin.tmpArr[e.Value] = arr[End.tmpArr[e.Key]];
                }
            }

            public override void Complete()
            {
                for (int idxE = 0; idxE < End.items.Length; ++idxE)
                {
                    End.Index2Array(idxE);
                    BcIndexes();
                    SliceIndexes();

                    int idxB = Begin.Array2Index();
                    End.items[idxE] = Begin.items[idxB];
                }
            }
        }

        public class IndexTableSlice<Type> : IndexTable<Type>
        {
            readonly int[][] indexes;

            public IndexTableSlice(NDArray<Type> nD, object[] args)
            {
                Begin = new NDArray<Type>(nD);
                var args0 = CompleteArgsAndReshape(Begin, args);
                indexes = Enumerable.Range(0, args0.Length).Select(k => GetSlice(Begin.Shape[k], args0[k].ToString())).ToArray();
                var nshape0 = indexes.Select(a => a.Count()).ToArray();
                End = new NDArray<Type>(nshape0);
            }

            public override IndexType indexType => IndexType.slice;

            public override void Complete()
            {
                for (int idxE = 0; idxE < End.items.Length; ++idxE)
                {
                    End.Index2Array(idxE);
                    for (int i = 0; i < End.Shape.Length; ++i)
                        Begin.tmpArr[i] = indexes[i][End.tmpArr[i]];

                    int idxB = Begin.Array2Index();
                    End.items[idxE] = Begin.items[idxB];
                }
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

        static object[] CompleteArgsAndReshape<Type>(NDArray<Type> nD0, object[] args)
        {
            int nbNewAxis = args.Count(i => i == null);
            if (nD0.Shape.Length + nbNewAxis < args.Length)
                throw new ArgumentException();

            List<int> nshape = nD0.Shape.ToList();
            for (int k = 0; k < args.Length; ++k)
            {
                var v = args[k];
                if (v == null)
                {
                    if (k >= nshape.Count) nshape.Add(1);
                    else nshape.Insert(k, 1);
                    args[k] = ":";
                }
                else if (v is int)
                {
                    args[k] = new NDArray<int>(new int[] { (int)v }, new int[] { 1 });
                }
            }

            nD0.ReShapeInplace(nshape.ToArray());

            List<object> args0 = args.ToList();
            if (nD0.Shape.Length > args.Length)
            {
                int diff = nD0.Shape.Length - args.Length;
                args0.AddRange(Enumerable.Repeat(":", diff));
            }

            return args0.ToArray();
        }

        public static void GetAtIndexes<Type>(this NDArray<Type> nD, params object[] args)
        {
            Console.WriteLine(nD.Shape.Glue(globalformat:"Initial Shape ({0})"));
            //if (args.All(o => o is string || o == null))
            //{
            //    var it = new IndexTableSlice<Type>(nD, args);
            //    it.Complete();
            //    Console.WriteLine(it.End);
            //}
            //else
            //{
            //    var it = new IndexTableNDarray<Type>(nD, args);
            //    it.Complete();
            //    Console.WriteLine(it.End);
            //}

            var indexManager = new IndexManager<Type>(nD, args);
            indexManager.Complete();
            Console.WriteLine(indexManager.End);
        }
    }
}
