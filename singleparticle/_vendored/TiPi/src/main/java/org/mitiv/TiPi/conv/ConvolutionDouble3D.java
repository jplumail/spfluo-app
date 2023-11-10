// *WARNING* This file has been automatically generated by TPP do not edit directly.
/*
 * This file is part of TiPi (a Toolkit for Inverse Problems and Imaging)
 * developed by the MitiV project.
 *
 * Copyright (c) 2014 the MiTiV project, http://mitiv.univ-lyon1.fr/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

package org.mitiv.TiPi.conv;

import org.mitiv.TiPi.base.Shape;
import org.mitiv.TiPi.utils.FFTUtils;
import org.mitiv.TiPi.linalg.shaped.ShapedVectorSpace;

import org.jtransforms.fft.DoubleFFT_3D;

/**
 * Implements FFT-based convolution for 3D arrays of double's.
 *
 * @author Éric Thiébaut
 */
class ConvolutionDouble3D extends ConvolutionDouble {

    /** FFT operator. */
    private DoubleFFT_3D fft = null;

    /** The operator R. */
    private final PushPullOperator R;

    /** The operator S. */
    private final PushPullOperator S;

    /** Number of element along 1st dimension of the work space. */
    private final int dim1;

    /** Number of element along 2nd dimension of the work space. */
    private final int dim2;

    /** Number of element along 3rd dimension of the work space. */
    private final int dim3;

    /**
     * Create a new convolution operator for 3D arrays of double's.
     *
     * <p> This protected constructor should not be directly used.  Call {@link
     * Convolution#build(Shape, ShapedVectorSpace, int[], ShapedVectorSpace,
     * int[])} instead. </p>
     *
     * @param wrk
     *        The dimensions of the work space. If {@code null}, the dimensions
     *        of the work space are automatically computed to be the smallest
     *        dimensions suitable for the FFT (see
     *        {@link FFTUtils#bestDimension(int)}) and large enough to encompass
     *        the input and output dimensions. If {@code wrk} is {@code null},
     *        it is probably better to left the offsets unspecified and set
     *        {@code inpOff} and {@code outOff} to be {@code null}.
     *
     * @param inp
     *        The input space.
     *
     * @param inpOff
     *        The position of the input region within the work space. If
     *        {@code null}, the input region is assumed to be centered;
     *        otherwise, it must have as many values as the rank of the input
     *        and output spaces of the operator.
     *
     * @param out
     *        The output space.
     *
     * @param outOff
     *        The position of the output region within the work space. If
     *        {@code null}, the output region assumed to be centered; otherwise,
     *        it must have as many values as the rank of the input and output
     *        spaces of the operator.
     *
     * @see Convolution#build(Shape, ShapedVectorSpace, int[],
     *      ShapedVectorSpace, int[])
     */
    public ConvolutionDouble3D(Shape wrk,
                                ShapedVectorSpace inp, int[] inpOff,
                                ShapedVectorSpace out, int[] outOff) {
        /* Initialize super class and check rank and dimensions (element type
           is checked by the super class constructor). */
        super(wrk, inp, inpOff, out, outOff);
        if (getRank() != 3) {
            throw new IllegalArgumentException("Input and output spaces must be 3D");
        }
      /*  //#     eval km1 = 3 - 1*/
        this.dim1 = workShape.dimension(0);
      /*  //#     eval km1 = 3 - 2*/
        this.dim2 = workShape.dimension(1);
      /*  //#     eval km1 = 3 - 3*/
        this.dim3 = workShape.dimension(2);
        this.R = new PushPullOperator(workShape, out.getShape(),
                                      outputOffsets, fastOutput);
        this.S = new PushPullOperator(workShape, inp.getShape(),
                                      inputOffsets, fastInput);
    }

    /** Create low-level FFT operator. */
    private final void createFFT() {
        if (fft == null) {
            fft = new DoubleFFT_3D(dim3,dim2,dim1);
        }
    }

    /** Apply in-place forward complex FFT. */
    @Override
    public final void forwardFFT(double z[]) {
        if (z.length != 2*getNumberOfFrequencies()) {
            throw new IllegalArgumentException("Bad argument size");
        }
        timerForFFT.resume();
        if (fft == null) {
            createFFT();
        }
        fft.complexForward(z);
        timerForFFT.stop();
    }

    /** Apply in-place backward complex FFT. */
    @Override
    public final void backwardFFT(double z[]) {
        if (z.length != 2*getNumberOfFrequencies()) {
            throw new IllegalArgumentException("Bad argument size");
        }
        timerForFFT.resume();
        if (fft == null) {
            createFFT();
        }
        fft.complexInverse(z, false);
        timerForFFT.stop();
    }

    @Override
    public void push(double z[], double x[], boolean adjoint) {
        if (adjoint) {
            R.push(z, x);
        } else {
            S.push(z, x);
        }
    }

    @Override
    public void pull(double x[], double z[], boolean adjoint) {
        if (adjoint) {
            S.pull(x, z);
        } else {
            R.pull(x, z);
        }
    }


    private class PushPullOperator {
        /** Internal and external spaces have the same dimensions. */
        private final boolean fast;

        /** Offset of region along 1st input dimension. */
        private final int off1;

        /** Offset of region along 2nd input dimension. */
        private final int off2;

        /** Offset of region along 3rd input dimension. */
        private final int off3;

        /** End of region along 1st input dimension. */
        private final int end1;

        /** End of region along 2nd input dimension. */
        private final int end2;

        /** End of region along 3rd input dimension. */
        private final int end3;

        /**
         * Create a real-complex push/pull operator.
         *
         * <p> A push/pull operator is in charge of exchanging the contents of
         * vectors between an internal space (the work space with complex
         * values) and an external space (the user space with real values). The
         * dimensions of the user space must be smaller or equal those of the
         * work space. This method assumes that the arguments are valid (in
         * fact they have been already checked by the constructors of the
         * {@link Convolution} class). </p>
         *
         * @param wrk
         *        The dimensions of the work space.
         *
         * @param usr
         *        The dimensions of the user space.
         *
         * @param off
         *        The offsets of the user space relative to the work space.
         *
         * @param fast
         *        True if user and work spaces have the same dimensions;
         *        false otherwise.
         */
        private PushPullOperator(Shape wrk, Shape usr, int[] off, boolean fast) {
            this.fast = fast;
            this.off1 = off[0];
            this.end1 = off1 + usr.dimension(0);
            this.off2 = off[1];
            this.end2 = off2 + usr.dimension(1);
            this.off3 = off[2];
            this.end3 = off3 + usr.dimension(2);
        }

        /** Set contents of work array. */
        private void push(final double[] z, final double[] x) {
            final double zero = 0;
            if (fast) {
                /* User and work spaces have the same size. */
                for (int j = 0, k = 0; j < x.length; ++j, k += 2) {
                    z[k] = x[j];
                    z[k+1] = zero;
                }
            } else {
                /* User space is smaller than work space. */
                int j = 0; // index in x array
                int k = 0; // index of real part in z array
                for (int i3 = 0; i3 < off3; ++i3) {
                    for (int i2 = 0; i2 < dim2; ++i2) {
                        for (int i1 = 0; i1 < dim1; ++i1, k += 2) {
                            z[k] = zero;
                            z[k+1] = zero;
                        }
                    }
                }
                for (int i3 = off3; i3 < end3; ++i3) {
                    for (int i2 = 0; i2 < off2; ++i2) {
                        for (int i1 = 0; i1 < dim1; ++i1, k += 2) {
                            z[k] = zero;
                            z[k+1] = zero;
                        }
                    }
                    for (int i2 = off2; i2 < end2; ++i2) {
                        for (int i1 = 0; i1 < off1; ++i1, k += 2) {
                            z[k] = zero;
                            z[k+1] = zero;
                        }
                        for (int i1 = off1; i1 < end1; ++i1, ++j, k += 2) {
                            z[k] = x[j];
                            z[k+1] = zero;
                        }
                        for (int i1 = end1; i1 < dim1; ++i1, k += 2) {
                            z[k] = zero;
                            z[k+1] = zero;
                        }
                    }
                    for (int i2 = end2; i2 < dim2; ++i2) {
                        for (int i1 = 0; i1 < dim1; ++i1, k += 2) {
                            z[k] = zero;
                            z[k+1] = zero;
                        }
                    }
                }
                for (int i3 = end3; i3 < dim3; ++i3) {
                    for (int i2 = 0; i2 < dim2; ++i2) {
                        for (int i1 = 0; i1 < dim1; ++i1, k += 2) {
                            z[k] = zero;
                            z[k+1] = zero;
                        }
                    }
                }
            }
        }

        /** Extract contents of work array. */
        private void pull(final double[] x, final double[] z) {
            if (fast) {
                /* User and work spaces have the same size. */
                for (int j = 0, k = 0; j < x.length; ++j, k += 2) {
                    x[j] = z[k];
                }
            } else {
                /* User space is smaller than work space. */
                int j = 0; // index in x array
                int k; // index of real part in z array
                for (int i3 = off3; i3 < end3; ++i3) {
                    for (int i2 = off2; i2 < end2; ++i2) {
                        k = (off1 + dim1*(i2 + dim2*i3))*2;
                        for (int i1 = off1; i1 < end1; ++i1, ++j, k += 2) {
                            x[j] = z[k];
                        }
                    }
                }
            }
        }
    }

}
