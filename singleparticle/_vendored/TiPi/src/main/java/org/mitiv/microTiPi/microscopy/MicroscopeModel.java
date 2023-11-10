/*
 * Copyright (c) 2017 Ferréol Soulez ferreol.soulez@univ-lyon1.fr
 *
 * This file is part of microTiPi
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

package org.mitiv.microTiPi.microscopy;

import org.mitiv.TiPi.array.Array3D;
import org.mitiv.TiPi.base.Shape;
import org.mitiv.TiPi.linalg.shaped.DoubleShapedVector;
import org.mitiv.TiPi.linalg.shaped.DoubleShapedVectorSpace;
import org.mitiv.TiPi.linalg.shaped.ShapedVector;
import org.mitiv.TiPi.linalg.shaped.ShapedVectorSpace;
import org.mitiv.TiPi.psf.PsfModel;
/**
 * Abstract class for to model PSF of any fluorescence microscope
 *
 * @author Ferréol
 *
 */
public abstract class MicroscopeModel extends PsfModel
{
    protected int PState=0;   // flag to prevent useless recomputation of the PSF
    protected final static boolean NORMALIZED = true;
    protected static final double DEUXPI = 2*Math.PI;
    protected double dxy; // the lateral pixel size in meter
    protected double dz; // the axial sampling step size in meter
    protected int Nx; // number of samples along lateral X-dimension
    protected int Ny; // number of samples along lateral Y-dimension
    protected int Nz; // number of samples along axial Z-dimension

    protected Array3D psf; //3D point spread function

    protected DoubleShapedVectorSpace[] parameterSpace;
    protected DoubleShapedVector[] parameterCoefs;

    /** Initialize the  PSF model containing parameters
     *  @param psfShape shape of the PSF array
     *  @param dxy lateral pixel size
     *  @param dz axial sampling step size
     *  @param single single precision flag
     */
    public MicroscopeModel(Shape psfShape,
            double dxy, double dz,
            boolean single)
    {
        super(psfShape, single);
        this.dxy = dxy;
        this.dz = dz;

        if (psfShape.rank() !=3){
            throw new IllegalArgumentException("Microscope PSF  should be 3D");
        }
        Nx = psfShape.dimension(0);
        Ny = psfShape.dimension(1);
        Nz = psfShape.dimension(2);
        this.psfShape = psfShape;
        this.setSingle(single);
    }



    /**
     * Apply the Jacobian to the gradient on the PSF to get the
     *  derivative with respect to the PSF parameters
     *
     * @param grad derivative with respect to the PSF pixels
     * @param xspace PSF parameter space
     * @return derivative with respect to the PSF parameters
     */
    abstract public DoubleShapedVector apply_Jacobian(ShapedVector grad, ShapedVectorSpace xspace);


    /**
     * @return return an array with the flags of each parameters
     */
    abstract public int[] getParametersFlags();



    /**
     *
     */
    protected abstract void computePsf();



}