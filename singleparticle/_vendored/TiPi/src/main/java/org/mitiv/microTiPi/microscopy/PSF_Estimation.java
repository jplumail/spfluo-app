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

import org.mitiv.TiPi.array.ShapedArray;
import org.mitiv.TiPi.base.Shape;
import org.mitiv.TiPi.conv.WeightedConvolutionCost;
import org.mitiv.TiPi.linalg.shaped.DoubleShapedVector;
import org.mitiv.TiPi.linalg.shaped.DoubleShapedVectorSpace;
import org.mitiv.TiPi.linalg.shaped.FloatShapedVectorSpace;
import org.mitiv.TiPi.linalg.shaped.ShapedVector;
import org.mitiv.TiPi.linalg.shaped.ShapedVectorSpace;
import org.mitiv.TiPi.optim.BoundProjector;
import org.mitiv.TiPi.optim.LineSearch;
import org.mitiv.TiPi.optim.MoreThuenteLineSearch;
import org.mitiv.TiPi.optim.OptimTask;
import org.mitiv.TiPi.optim.ReverseCommunicationOptimizer;
import org.mitiv.TiPi.optim.VMLMB;
import org.mitiv.TiPi.array.ArrayUtils;

/**
 * Define a class for PSF estimation. This class contains all the parameters
 * needed to estimate the PSF defined in the pupil property.
 * The fit is performed using the fitPSF function.
 *
 *
 * @author Ferréol
 *
 */
public class PSF_Estimation  {

    private double gatol = 0.0;
    private double grtol = 1e-3;
    private int limitedMemorySize = 5;
    private double lowerBound = Double.NEGATIVE_INFINITY;
    private double upperBound = Double.POSITIVE_INFINITY;
    private int maxiter = 20;
    private int maxeval = 20;
    private ShapedArray data = null;
    private ShapedArray obj = null;
    //   private ShapedArray psf = null;
    private double fcost = 0.0;
    private ShapedVector gcost = null;
    private MicroscopeModel pupil = null;
    private ReverseCommunicationOptimizer minimizer = null;
    private  ShapedArray weights = null;
    private boolean single;

    private boolean run = true;


    private boolean debug = false;



    /**
     * Build a PSF_Estimation object for the PSF defined in pupil
     * @param pupil
     */
    public PSF_Estimation(MicroscopeModel pupil) {
        if (pupil!=null){
            this.pupil = pupil;
            single = pupil.isSingle();
        }else{
            fatal("pupil not specified");
        }
    }

    /**
     * @param positivity
     */
    public void enablePositivity(Boolean positivity) {
        setLowerBound(positivity ? 0.0 : Double.NEGATIVE_INFINITY);
    }
    private static void fatal(String reason) {
        throw new IllegalArgumentException(reason);
    }


    /**
     * Perform the PSF estimation on the parameters indexed by flag
     * @param flag
     */
    public void fitPSF(  int flag) {
        run =true;
        // FIXME set a best X
        DoubleShapedVector x = null;
        double best_cost = Double.POSITIVE_INFINITY;
        // Check input data and get dimensions.
        if (data == null) {
            fatal("Input data not specified.");
        }


        x = pupil.parameterCoefs[flag];



        Shape dataShape = data.getShape();
        int rank = data.getRank();
        ShapedVectorSpace dataSpace, objSpace;

        DoubleShapedVector best_x = x.clone();
        // Check the PSF.
        if (obj == null) {
            fatal("Object not specified.");
        }
        if (obj.getRank() != rank) {
            fatal("Obj must have same rank as data.");
        }

        if(single){
            dataSpace = new FloatShapedVectorSpace(dataShape);
            objSpace = new FloatShapedVectorSpace(dataShape);
        }else{
            dataSpace = new DoubleShapedVectorSpace(dataShape);
            objSpace = new DoubleShapedVectorSpace(dataShape);
        }

        // Initialize a vector space and populate it with workspace vectors.

        DoubleShapedVectorSpace variableSpace = x.getSpace();
        int[] off ={0,0, 0};
        // Build convolution operator.
        WeightedConvolutionCost fdata = WeightedConvolutionCost.build(objSpace, dataSpace);
        fdata.setPSF(obj,off);
        fdata.setData(data);
        fdata.setWeights(weights,true);

        if (debug) {
            System.out.println("Vector space initialization complete.");
        }

        gcost = objSpace.create();
        fcost = fdata.computeCostAndGradient(1.0, objSpace.create(pupil.getPsf() ), gcost, true);
        best_cost = fcost;
        best_x = x.clone();

        if (debug) {
            System.out.println("Cost function initialization complete.");
        }

        // Initialize the non linear conjugate gradient
        LineSearch lineSearch = null;
        VMLMB vmlmb = null;
        BoundProjector projector = null;
        int bounded = 0;
        limitedMemorySize = 0;

        if (lowerBound != Double.NEGATIVE_INFINITY) {
            bounded |= 1;
        }
        if (upperBound != Double.POSITIVE_INFINITY) {
            bounded |= 2;
        }


        if (debug) {
            System.out.println("bounded");
            System.out.println(bounded);
        }

        /* No bounds have been specified. */
        lineSearch = new MoreThuenteLineSearch(0.05, 0.1, 1E-17);

        int m = (limitedMemorySize > 1 ? limitedMemorySize : 5);
        vmlmb = new VMLMB(variableSpace, projector, m, lineSearch);
        vmlmb.setAbsoluteTolerance(gatol);
        vmlmb.setRelativeTolerance(grtol);
        minimizer = vmlmb;

        if (debug) {
            System.out.println("Optimization method initialization complete.");
        }

        DoubleShapedVector gX = variableSpace.create();
        OptimTask task = minimizer.start();
        while (run) {
            if (task == OptimTask.COMPUTE_FG) {
                pupil.setParam(x);

                pupil.computePsf();

                fcost = fdata.computeCostAndGradient(1.0, objSpace.create(pupil.getPsf()), gcost, true);

                if(fcost<best_cost){
                    best_cost = fcost;
                    best_x = x.clone();


                    if(debug){
                        System.out.println("Cost: " + best_cost);
                    }
                }
                gX =  pupil.apply_Jacobian(gcost,x.getSpace());

            } else if (task == OptimTask.NEW_X || task == OptimTask.FINAL_X) {
                boolean stop = (task == OptimTask.FINAL_X);
                if (! stop && maxiter >= 0 && minimizer.getIterations() >= maxiter) {
                    if (debug){
                        System.out.format("Warning: too many iterations (%d).\n", maxiter);
                    }
                    stop = true;
                }
                if (stop) {
                    break;
                }
            } else {
                if (debug){
                    System.out.println("TiPi: PSF_Estimation, "+task+" : "+minimizer.getReason());
                }
                break;
            }
            if (debug) {
                System.out.println("Evaluations");
                System.out.println(minimizer.getEvaluations());
                System.out.println("Iterations");
                System.out.println(minimizer.getIterations());
            }

            if (minimizer.getEvaluations() >= maxeval) {
                if( debug){
                    System.out.format("Warning: too many evaluation (%d).\n", maxeval);
                }
                break;
            }
            task = minimizer.iterate(x, fcost, gX);

        }


        pupil.setParam(best_x);

    }

    /* Below are all methods required for a ReconstructionJob. */

    /** Enable debugging mode
     * @param value
     */
    public void setDebugMode(boolean value) {
        debug = value;
    }

    /** Fix the maximum number of iteration
     * @param value
     */
    public void setMaximumIterations(int value) {
        maxiter = value;
        maxeval = 2* value; // 2 or 3 times more evaluations than iterations seems reasonable
    }

    /** Set the number of steps kept in memory for hessian estimation.
     * @param value
     */
    public void setLimitedMemorySize(int value) {
        limitedMemorySize = value;
    }

    /**
     * @param value
     */
    public void setAbsoluteTolerance(double value) {
        gatol = value;
    }

    /**
     * @param value
     */
    public void setRelativeTolerance(double value) {
        grtol = value;
    }

    /**
     * @param value
     */
    public void setLowerBound(double value) {
        lowerBound = value;
    }

    /**
     * @param value
     */
    public void setUpperBound(double value) {
        upperBound = value;
    }

    /**
     * Emergency stop
     */
    public void abort(){
        run = false;
    }



    /** Set the weights (inverse covariance matrix)
     * @param shapedArray
     */
    public void setWeight(ShapedArray wgtArray){
        this.weights = ArrayUtils.pad(wgtArray,pupil.getShape());
    }

    /** Change the microscope model
     * @param pupil
     */
    public void setPupil(MicroscopeModel pupil) {
        this.pupil = pupil;
    }

    /**
     * @return the pupil
     */
    public MicroscopeModel getPupil() {
        return pupil;
    }

    /**
     * @return the data
     */
    public ShapedArray getData() {
        return data;
    }

    /**
     * @param shapedArray
     */
    public void setData(ShapedArray shapedArray) {
        this.data = shapedArray;
    }

    /**
     * @return the PSF
     */
    public ShapedArray getPsf() {
        return pupil.getPsf();
    }

    /**
     * @return the current number of iteration
     */
    public int getIterations() {
        return (minimizer == null ? 0 : minimizer.getIterations());
    }

    /**
     * @return the current number of evaluation
     */
    public int getEvaluations() {
        return (minimizer == null ? 0 : minimizer.getEvaluations());
    }


    /**
     * @return the value of the cost function
     */
    public double getCost() {
        return fcost;
    }

    /**
     * @param objArray
     */
    public void setObj(ShapedArray objArray) {
        this.obj = objArray;

    }

    /** Return the object containing the full description of the PSF
     * @return pupil
     */
    public MicroscopeModel getModel(){
        return this.pupil;
    }

    /**
     * Free some memory
     */
    public void freeMem(){
        pupil.freeMem();
    }
}
