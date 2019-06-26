using System;
using System.Collections.Generic;
using System.Linq;

namespace DesertLandCNN
{
    public class Conv2d<Type> : Layer<Type>
    {
        private NDArray<Type> W, w0, layerInput, X_col, W_col;
        private IOptimizer<Type> WOpt, w0Opt;

        public override int Parameters => NumDN.ShapeLength(W.Shape) + NumDN.ShapeLength(w0.Shape);

        public override int[] GetOutputShape() => outputShape();

        private int numFilters, stride, fWidth, fHeight, channel;
        private int[] filterShape;
        private string padding;

        public Conv2d(int numFilters, int[] filterShape, int[] inputShape, int stride, string padding)
        {
            this.Name = "Conv2d";
            this.numFilters = numFilters;
            this.stride = stride;
            Inputs = inputShape.ToArray();
            this.filterShape = filterShape.ToArray();
            this.padding = padding;
        }

        public Conv2d(int numFilters, int[] filterShape, int stride, string padding)
        {
            this.Name = "Conv2d";
            this.numFilters = numFilters;
            this.stride = stride;
            this.filterShape = filterShape.ToArray();
            this.padding = padding;
        }

        public override void Initialize(IOptimizer<Type> optimizer = null)
        {
            fHeight = filterShape[0];
            fWidth = filterShape[1];
            channel = Inputs[0];
            double lim = 1.0 / Math.Sqrt(NumDN.ShapeLength(filterShape));
            W = NumDN.Uniform<Type>(-lim, lim, numFilters, channel, fWidth, fHeight);
            w0 = NDArray<Type>.Zeros(numFilters, 1);

            WOpt = optimizer.Clone();
            w0Opt = optimizer.Clone();
        }

        /*
                batch_size, channels, height, width = X.shape
                self.layer_input = X
                # Turn image shape into column shape
                # (enables dot product between input and weights)
                self.X_col = image_to_column(X, self.filter_shape, stride=self.stride, output_shape=self.padding)
                # Turn weights into column shape
                self.W_col = self.W.reshape((self.n_filters, -1))
                # Calculate output
                output = self.W_col.dot(self.X_col) + self.w0
                # Reshape into (n_filters, out_height, out_width, batch_size)
                output = output.reshape(self.output_shape() + (batch_size, ))
                # Redistribute axises so that batch size comes first
                return output.transpose(3,0,1,2)
         */
        public override NDArray<Type> Forward(NDArray<Type> X, bool isTraining)
        {
            int batchSize = X.Shape[0];
            int chan = X.Shape[1];
            int height = X.Shape[2];
            int width = X.Shape[3];

            layerInput = new NDArray<Type>(X);
            X_col = img2col(X, filterShape, stride, padding);
            W_col = W.ReShape(numFilters, -1);
            var output = NDArray<Type>.Dot(W_col, X_col) + w0;
            output.ReShapeInplace(outputShape(batchSize));

            return output.transpose(3, 0, 1, 2);
        }

        /*
        def backward_pass(self, accum_grad):
            # Reshape accumulated gradient into column shape
            accum_grad = accum_grad.transpose(1, 2, 3, 0).reshape(self.n_filters, -1)

            if self.trainable:
                # Take dot product between column shaped accum. gradient and column shape
                # layer input to determine the gradient at the layer with respect to layer weights
                grad_w = accum_grad.dot(self.X_col.T).reshape(self.W.shape)
                # The gradient with respect to bias terms is the sum similarly to in Dense layer
                grad_w0 = np.sum(accum_grad, axis=1, keepdims=True)

                # Update the layers weights
                self.W = self.W_opt.update(self.W, grad_w)
                self.w0 = self.w0_opt.update(self.w0, grad_w0)

            # Recalculate the gradient which will be propogated back to prev. layer
            accum_grad = self.W_col.T.dot(accum_grad)
            # Reshape from column shape to image shape
            accum_grad = column_to_image(accum_grad,
                                    self.layer_input.shape,
                                    self.filter_shape,
                                    stride=self.stride,
                                    output_shape=self.padding)

            return accum_grad
         */
        public override NDArray<Type> Backward(NDArray<Type> accumGrad)
        {
            var accumGrad0 = accumGrad.transpose(1, 2, 3, 0).ReShape(numFilters, -1);
            if (trainable)
            {
                var gradW = NDArray<Type>.Dot(accumGrad0, X_col.T).ReShape(W.Shape);
                var gradw0 = NumDN.Sum(accumGrad0, true, 1);
                W = WOpt.Update(W, gradW);
                w0 = w0Opt.Update(w0, gradw0);
            }

            accumGrad0 = NDArray<Type>.Dot(W_col.T, accumGrad0);
            accumGrad0 = col2img(accumGrad0, layerInput.Shape, filterShape, stride, padding);

            return accumGrad0;
        }

        public override void SetInputShape(int[] inputShape)
        {
            Inputs = inputShape.ToArray();
        }

        /*
        def output_shape(self):
            channels, height, width = self.input_shape
            pad_h, pad_w = determine_padding(self.filter_shape, output_shape=self.padding)
            output_height = (height + np.sum(pad_h) - self.filter_shape[0]) / self.stride + 1
            output_width = (width + np.sum(pad_w) - self.filter_shape[1]) / self.stride + 1
            return self.n_filters, int(output_height), int(output_width)
         */
        int[] outputShape()
        {
            (int chan, int h, int w) = (Inputs[0], Inputs[1], Inputs[2]);
            (var ph, var pw) = determinePadding(filterShape, padding);
            int oh = (int)((double)(h + ph.Item1 + ph.Item2 - filterShape[0]) / stride + 1);
            int ow = (int)((double)(w + pw.Item1 + pw.Item2 - filterShape[1]) / stride + 1);

            return new int[] { numFilters, oh, ow };
        }

        int[] outputShape(int batchSize)
        {
            (int chan, int h, int w) = (Inputs[0], Inputs[1], Inputs[2]);
            (var ph, var pw) = determinePadding(filterShape, padding);
            int oh = (int)((double)(h + ph.Item1 + ph.Item2 - filterShape[0]) / stride + 1);
            int ow = (int)((double)(w + pw.Item1 + pw.Item2 - filterShape[1]) / stride + 1);

            return new int[] { numFilters, oh, ow, batchSize };
        }

        /*
            # Method which calculates the padding based on the specified output shape and the
            # shape of the filters
            def determine_padding(filter_shape, output_shape="same"):

                # No padding
                if output_shape == "valid":
                    return (0, 0), (0, 0)
                # Pad so that the output shape is the same as input shape (given that stride=1)
                elif output_shape == "same":
                    filter_height, filter_width = filter_shape

                    # Derived from:
                    # output_height = (height + pad_h - filter_height) / stride + 1
                    # In this case output_height = height and stride = 1. This gives the
                    # expression for the padding below.
                    pad_h1 = int(math.floor((filter_height - 1)/2))
                    pad_h2 = int(math.ceil((filter_height - 1)/2))
                    pad_w1 = int(math.floor((filter_width - 1)/2))
                    pad_w2 = int(math.ceil((filter_width - 1)/2))

                    return (pad_h1, pad_h2), (pad_w1, pad_w2)
         */
        public static ((int,int),(int,int)) determinePadding(int[] fShape, string outShape = "same")
        {
            if (outShape == "valid")
                return ((0, 0), (0, 0));

            int h = fShape[0];
            int w = fShape[1];
            int ph1 = (int)Math.Floor((h - 1.0) / 2.0);
            int ph2 = (int)Math.Ceiling((h - 1.0) / 2.0);
            int pw1 = (int)Math.Floor((w - 1.0) / 2.0);
            int pw2 = (int)Math.Ceiling((w - 1.0) / 2.0);

            return ((ph1, ph2), (pw1, pw2));
        }

        /*

            # Reference: CS231n Stanford
            def get_im2col_indices(images_shape, filter_shape, padding, stride=1):
                # First figure out what the size of the output should be
                batch_size, channels, height, width = images_shape
                filter_height, filter_width = filter_shape
                pad_h, pad_w = padding
                out_height = int((height + np.sum(pad_h) - filter_height) / stride + 1)
                out_width = int((width + np.sum(pad_w) - filter_width) / stride + 1)

                i0 = np.repeat(np.arange(filter_height), filter_width)
                i0 = np.tile(i0, channels)
                i1 = stride * np.repeat(np.arange(out_height), out_width)
                j0 = np.tile(np.arange(filter_width), filter_height * channels)
                j1 = stride * np.tile(np.arange(out_width), out_height)
                i = i0.reshape(-1, 1) + i1.reshape(1, -1)
                j = j0.reshape(-1, 1) + j1.reshape(1, -1)

                k = np.repeat(np.arange(channels), filter_height * filter_width).reshape(-1, 1)

                return (k, i, j)
         */
        public static (NDArray<int>, NDArray<int>, NDArray<int>) img2colIndices(int[] iShape, int[] fShape, int stride, ((int, int), (int, int)) padding)
        {
            double stride0 = stride;
            int batch = iShape[0];
            int chan = iShape[1];
            int h = iShape[2];
            int w = iShape[3];
            int fh = fShape[0];
            int fw = fShape[1];
            (var ph, var pw) = padding;
            int oh = (int)((h + ph.Item1 + ph.Item2 - fh) / stride0 + 1);
            int ow = (int)((w + pw.Item1 + pw.Item2 - fw) / stride0 + 1);

            var i0 = NumDN.Repeat(NumDN.ARange(fh), fw);
            i0 = NumDN.Tile(i0, chan);
            var i1 = NumDN.Repeat(NumDN.ARange(oh), ow).Select(a => a * stride).ToArray();
            var j0 = NumDN.Tile(NumDN.ARange(fw), fh * chan);
            var j1 = NumDN.Tile(NumDN.ARange(ow), oh);

            var i = new NDArray<int>(i0, new int[] { i0.Length, 1 }) + new NDArray<int>(i1, new int[] { 1, i1.Length });
            var j = new NDArray<int>(j0, new int[] { j0.Length, 1 }) + new NDArray<int>(j1, new int[] { 1, j1.Length });

            var k0 = NumDN.Repeat(NumDN.ARange(chan), fh * fw);
            var k = new NDArray<int>(k0, new int[] { k0.Length, 1 });

            return (k, i, j);
        }

        /*
        # Method which turns the image shaped input to column shape.
        # Used during the forward pass.
        # Reference: CS231n Stanford
        def image_to_column(images, filter_shape, stride, output_shape='same'):
            filter_height, filter_width = filter_shape

            pad_h, pad_w = determine_padding(filter_shape, output_shape)

            # Add padding to the image
            images_padded = np.pad(images, ((0, 0), (0, 0), pad_h, pad_w), mode='constant')

            # Calculate the indices where the dot products are to be applied between weights
            # and the image
            k, i, j = get_im2col_indices(images.shape, filter_shape, (pad_h, pad_w), stride)

            # Get content from image at those indices
            cols = images_padded[:, k, i, j]
            channels = images.shape[1]
            # Reshape content into column shape
            cols = cols.transpose(1, 2, 0).reshape(filter_height * filter_width * channels, -1)
            return cols
         
         */
        public static NDArray<Type> img2col(NDArray<Type> images, int[] fShape, int stride, string outShape = "same")
        {
            (var fh, var fw) = (fShape[0], fShape[1]);
            (var ph, var pw) = determinePadding(fShape, outShape);
            var imgsPadded = NumDN.Pad(images, (0, 0), (0, 0), ph, pw);
            (var k, var i, var j) = img2colIndices(images.Shape, fShape, stride, (ph, pw));

            var cols = imgsPadded.GetIndexes(":", k, i, j);
            int chan = images.Shape[1];
            cols = cols.transpose(1, 2, 0).ReShape(fh * fw * chan, -1);

            return cols;
        }

        /*
            # Method which turns the column shaped input to image shape.
            # Used during the backward pass.
            # Reference: CS231n Stanford
            def column_to_image(cols, images_shape, filter_shape, stride, output_shape='same'):
                batch_size, channels, height, width = images_shape
                pad_h, pad_w = determine_padding(filter_shape, output_shape)
                height_padded = height + np.sum(pad_h)
                width_padded = width + np.sum(pad_w)
                images_padded = np.empty((batch_size, channels, height_padded, width_padded))

                # Calculate the indices where the dot products are applied between weights
                # and the image
                k, i, j = get_im2col_indices(images_shape, filter_shape, (pad_h, pad_w), stride)

                cols = cols.reshape(channels * np.prod(filter_shape), -1, batch_size)
                cols = cols.transpose(2, 0, 1)
                # Add column content to the images at the indices
                np.add.at(images_padded, (slice(None), k, i, j), cols)

                # Return image without padding
                return images_padded[:, :, pad_h[0]:height+pad_h[0], pad_w[0]:width+pad_w[0]]
         */
        public static NDArray<Type> col2img(NDArray<Type> cols, int[] imagesShape, int[] fShape, int stride, string outShape = "same")
        {
            (int batchSize, int chan, int h, int w) = (imagesShape[0], imagesShape[1], imagesShape[2], imagesShape[3]);
            (var ph, var pw) = determinePadding(fShape, outShape);
            int hp = h + ph.Item1 + ph.Item2;
            int wp = w + pw.Item1 + pw.Item2;
            var imgsPadded = new NDArray<Type>(batchSize, chan, hp, wp);
            //var imgsPadded = NumDN.Uniform<Type>(1e-5, 1 - 1e-5, batchSize, chan, hp, wp);

            (var k, var i, var j) = img2colIndices(imagesShape, fShape, stride, (ph, pw));
            var cols0 = cols.ReShape(chan * NumDN.ShapeLength(fShape), -1, batchSize);
            cols0 = cols0.transpose(2, 0, 1);
            var imgsPadded2 = ExtensionIndexes.AddAt(imgsPadded, cols0).Indexes(":", k, i, j);

            return imgsPadded2.GetIndexes(":", ":", $"{ph.Item1}:{h + ph.Item1}", $"{pw.Item1}:{w + pw.Item1}");
        }
    }
}
