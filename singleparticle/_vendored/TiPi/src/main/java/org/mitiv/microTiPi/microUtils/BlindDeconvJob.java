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

package org.mitiv.microTiPi.microUtils;

import org.mitiv.microTiPi.microscopy.MicroscopeModel;
import org.mitiv.microTiPi.microscopy.PSF_Estimation;

import org.mitiv.TiPi.array.ArrayUtils;
import org.mitiv.TiPi.array.ShapedArray;
import org.mitiv.TiPi.jobs.DeconvolutionJob;
import org.mitiv.TiPi.weights.WeightUpdater;

/**
 * BlindDeconvJob is the solver for blind deconvolution
 *
 * @author ferreol
 *
 */
public class BlindDeconvJob {

    private int totalNbOfBlindDecLoop;
    private ShapedArray psfArray;
    private boolean debug=false;
    //   private ShapedArray objArray;
    private PSF_Estimation psfEstimation;
    private DeconvolutionJob deconvolver;
    /**
     * @return the deconvolver
     */
    public DeconvolutionJob getDeconvolver() {
        return deconvolver;
    }

    private int[] parametersFlags;
    private boolean run =false;
    private int[] maxIter;
    private WeightUpdater wghtUpdt=null;

    /**
     * Build the solver for blind deconvolution
     *
     * @param totalNbOfBlindDecLoop
     *        number of loop
     * @param parametersFlags
     *        array of flags indicating which parameters of the MicroscopeModel
     *        will be estimated
     * @param maxIter
     *        number of iterations for each PSF sub-problems.
     *        It is has the same size as parametersFlags
     * @param psfEstimation
     *        solver for the PSF sub-problem. It contains the MicroscopeModel object
     *        that describes the PSF parameterization
     * @param deconvolver
     *         solver for the object sub-problem
     * @param debug
     *         debug flag
     *
     */
    public BlindDeconvJob(int totalNbOfBlindDecLoop,int[] parametersFlags,int[] maxIter, PSF_Estimation psfEstimation ,DeconvolutionJob deconvolver,WeightUpdater wghtUpdt, boolean debug ) {
        this.totalNbOfBlindDecLoop = totalNbOfBlindDecLoop;
        this.parametersFlags = parametersFlags;
        this.maxIter = maxIter;
        this.psfEstimation = psfEstimation;
        this.deconvolver = deconvolver;
        this.debug = debug;
        this.wghtUpdt =wghtUpdt;
    }

    /**
     * Launch the blind deconvolution algorithm
     * @param objArray
     *        initial guess of the object
     * @return estimated object
     *
     */
    public ShapedArray blindDeconv(ShapedArray objArray){
        run =true;
        for(int i = 0; i < totalNbOfBlindDecLoop; i++) {
            psfArray = ArrayUtils.roll(psfEstimation.getPupil().getPsf());
            psfEstimation.freeMem();

            deconvolver.updatePsf(psfArray);

          /*  if(wghtUpdt!=null) {
                wghtUpdt.update(deconvolver);
            }*/
            objArray = deconvolver.deconv(objArray);
            if(wghtUpdt!=null) {
                psfEstimation.setWeight(wghtUpdt.update(deconvolver));
            }
            //Emergency stop
            if (!run) {
                return objArray;
            }
            if (i<totalNbOfBlindDecLoop-1) {
                psfEstimation.setObj(objArray);
                for (int j = 0; j < parametersFlags.length; j++) {
                    if (debug ) {
                        System.out.println("------------------");
                        System.out.println("  "+ j+" estimation");
                        System.out.println("------------------");
                    }
                    psfEstimation.setRelativeTolerance(0.);
                    psfEstimation.setMaximumIterations(maxIter[j]);
                    if(maxIter[j]>0){
                        psfEstimation.fitPSF( parametersFlags[j]);
                    }
                    //Emergency stop
                    if (!run) {
                        return objArray;
                    }
                }
            }
        }
        run = false;
        return objArray;
    }

    /**
     * Check whether the blind deconvolution is running
     * @return run
     */
    public boolean isRunning() {
        return run;
    }

    /**
     * Emergency stop
     */
    public void abort(){
        System.out.println("abort");
        run  = false;
        deconvolver.abort();
        psfEstimation.abort();
    }

    /**
     * Return the current PSF
     * @return psf
     */
    public ShapedArray getPsf() {
        return psfArray;
    }

    /**
     * Return the current pupil object
     * @return pupil
     */
    public  MicroscopeModel getPupil() {
        return psfEstimation.getPupil();
    }

    /**
     * Return the convolved object (model)
     * @return model
     */
    public ShapedArray getModel() {
        return deconvolver.getModel();
    }
}
