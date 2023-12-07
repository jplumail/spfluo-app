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

package commands;

import java.io.IOException;
import java.io.PrintStream;
import java.util.List;
import java.util.Locale;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

import org.mitiv.TiPi.array.ArrayUtils;
import org.mitiv.TiPi.array.ShapedArray;
import org.mitiv.TiPi.base.Shape;
import org.mitiv.TiPi.invpb.EdgePreservingDeconvolution;
import org.mitiv.TiPi.optim.OptimTask;
import org.mitiv.microTiPi.epifluorescence.WideFieldModel;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.formats.FormatException;


public class EdgePreservingDeconvolutionCommand {
    private PrintStream stream = System.out;

    @Option(name = "-init", usage = "Name of initial image file.", metaVar = "INIT")
    private String initName = null;

    @Option(name = "-psf", usage = "Name of point spread function file.", metaVar = "FILENAME")
    private String psfName = null;

    // WidefieldModel args
    @Option(name = "-nPhase", usage = "Number of zernike describing the pupil phase", metaVar = "N")
    private int nPhase = 19;

    @Option(name = "-nModulus", usage = "Number of zernike describing the pupil modulus", metaVar = "N")
    private int nModulus = 0;

    @Option(name = "-NA", usage = "Numerical aperture", metaVar = "NA")
    private double NA = 1.4;

    @Option(name = "-lambda", usage = "Wavelength in nm", metaVar = "lambda")
    private double lambda = 500;

    @Option(name = "-ni", usage = "Refractive index", metaVar = "ni")
    private double ni = 1.518;

    @Option(name = "-dxy", usage = "Lateral pixel size in nm", metaVar = "dxy")
    private double dxy = 1.;

    @Option(name = "-dz", usage = "Axial pixel size in nm", metaVar = "dz")
    private double dz = 1.;

    @Option(name = "-radial", usage = "Radial option")
    private boolean radial;

    @Option(name = "-normalize", usage = "Normalize the point spread function.")
    private boolean normalizePSF = false;

    @Option(name = "-weights", usage = "Name statistical weights file.", metaVar = "FILENAME")
    private String weightsName = null;

    @Option(name = "-noise", usage = "Standard deviation of the noise.", metaVar = "SIGMA")
    private double sigma = Double.NaN;

    @Option(name = "-gain", usage = "Detector gain.", metaVar = "GAMMA")
    private double gamma = Double.NaN;

    @Option(name = "-invalid", usage = "Name of invalid data file.", metaVar = "FILENAME")
    private String invalidName = null;

    @Option(name = "-mu", usage = "Regularization level.", metaVar = "MU")
    private double mu = 10.0;

    @Option(name = "-epsilon", usage = "Edge threshold.", metaVar = "EPSILON")
    private double epsilon = 1.0;

    @Option(name = "-gatol", usage = "Absolute gradient tolerance for the convergence.", metaVar = "GATOL")
    private double gatol = 0.0;

    @Option(name = "-grtol", usage = "Relative gradient tolerance for the convergence.", metaVar = "GRTOL")
    private double grtol = 1e-3;

    @Option(name = "-mem", usage = "If M > 0, use quasi-Newton method with M previous steps; otherwise, use non-linear conjugate gradient.", metaVar = "M")
    private int limitedMemorySize = 5;

    @Option(name = "-min", usage = "Lower bound for the variables.", metaVar = "LOWER")
    private double lowerBound = Double.NEGATIVE_INFINITY;

    @Option(name = "-max", usage = "Upper bound for the variables.", metaVar = "UPPER")
    private double upperBound = Double.POSITIVE_INFINITY;

    @Option(name = "-single", usage = "Force single precision.")
    private boolean single = false;

    @Option(name = "-help", aliases = {"--help", "-h", "-?"}, usage = "Display help.")
    private boolean help;

    @Option(name = "-verbose", usage = "Verbose mode.")
    private boolean verbose = false;

    @Option(name = "-debug", usage = "Debug mode.")
    private boolean debug = false;

    @Option(name = "-maxiter", usage = "Maximum number of iterations, -1 for no limits.")
    private int maxiter = 200;

    @Option(name = "-maxeval", usage = "Maximum number of evaluations, -1 for no limits.")
    private int maxeval = -1;

    @Option(name = "-pad", usage = "Padding method.", metaVar = "\"auto\"|\"min\"|NUMBER")
    private String paddingMethod = "auto";

    @Option(name = "-fill", usage = "Value for padding.", metaVar = "VALUE")
    private double fillValue = Double.NaN;

    @Option(name = "-crop", usage = "Crop result to same size as input.")
    private boolean crop = false;

    @Argument
    private List<String> arguments;

    static private void usage(CmdLineParser parser, int code) {
        PrintStream stream = (code == 0 ? System.out : System.err);
        stream.println("Usage: deconv [OPTIONS] INPUT OUTPUT");
        if (code == 0) {
            stream.println("Options:");
            parser.getProperties().withUsageWidth(80);
            parser.printUsage(stream);
        } else {
            stream.println("Try option -help for a more complete description of options.");
        }
        System.exit(code);
    }

    public static void main(String[] args) throws DependencyException, ServiceException, FormatException, IOException {

        // Switch to "US" locale to avoid problems with number formats.
        Locale.setDefault(Locale.US);

        // Parse options.
        EdgePreservingDeconvolutionCommand job = new EdgePreservingDeconvolutionCommand();
        CmdLineParser parser = new CmdLineParser(job);
        try {
            parser.parseArgument(args);
        } catch (CmdLineException e) {
            System.err.format("Error: %s\n", e.getMessage());
            usage(parser, 1);
        }
        if (job.help) {
            usage(parser, 0);
        }

        // Deal with remaining arguments.
        int size = (job.arguments == null ? 0 : job.arguments.size());
        if (size != 2) {
            System.err.format("Too %s arguments.\n", (size < 2 ? "few" : "many"));
            usage(parser, 1);
        }
        String inputName = job.arguments.get(0);
        String outputName = job.arguments.get(1);

        EdgePreservingDeconvolution solver = new EdgePreservingDeconvolution();

        try {
            // Read the blurred data and the PSF.
            solver.setForceSinglePrecision(job.single);
            solver.setData(MainCommand.loadData(inputName, job.single));
            if (job.psfName != null) {
                solver.setPSF(MainCommand.loadData(job.psfName, job.single), job.normalizePSF);
            } else {
                Shape shape = solver.getData().getShape();
                int Nxy = Math.min(shape.dimension(0), shape.dimension(1));
                int[] psfDims = {Nxy, Nxy, shape.dimension(2)};
                Shape psfShape = new Shape(psfDims);
                WideFieldModel pupil = new WideFieldModel(psfShape, job.nPhase, job.nModulus, job.NA, job.lambda*1E-9, job.ni, job.dxy*1E-9, job.dz*1E-9, job.radial, job.single);
                solver.setPSF(ArrayUtils.roll(pupil.getPsf()), job.normalizePSF);
            }

            // Deal with the weights.
            System.err.format("sigma = %g, gamma = %g\n", job.sigma, job.gamma);
            if (job.weightsName != null) {
                if (! isnan(job.sigma) || ! isnan(job.gamma)) {
                    System.err.println("Warning: options `-gain` and `-noise` are ignored when `-weights` is specified.");
                }
                solver.setWeights(MainCommand.loadData(job.weightsName, job.single));
            } else {
                if (isnan(job.sigma) && ! isnan(job.gamma)) {
                    System.err.println("Warning: option `-gain` alone is ignored, use it with `-noise`.");
                }
                solver.setDetectorNoise(job.sigma);
                solver.setDetectorGain(job.gamma);
            }

            // Deal with bad pixels.
            if (job.invalidName != null) {
                // FIXME: there should be a way to load a mask (i.e. as a boolean array)
                solver.setBads(MainCommand.loadData(job.invalidName, job.single));
            }

            // Compute dimensions of result.
            Shape dataShape = solver.getData().getShape();
            Shape psfShape = solver.getPSF().getShape();
            int[] objDims = MainCommand.getPaddingShape(job.paddingMethod, dataShape, psfShape);
            solver.setObjectShape(objDims);
            solver.setFillValue(job.fillValue);

            // Result and initial solution.
            if (job.initName != null) {
                solver.setInitialSolution(MainCommand.loadData(job.initName, job.single));
            }

            solver.setAbsoluteTolerance(job.gatol);
            solver.setRelativeTolerance(job.grtol);
            solver.setLowerBound(job.lowerBound);
            solver.setUpperBound(job.upperBound);
            solver.setLimitedMemorySize(Math.max(0, job.limitedMemorySize));
            solver.setRegularizationLevel(job.mu);
            solver.setEdgeThreshold(job.epsilon);
            solver.setMaximumIterations(job.maxiter);
            solver.setMaximumEvaluations(job.maxeval);
            solver.setDebug(job.debug);
            solver.setSaveBest(true);

            OptimTask task = solver.start();
             while (true) {
                 if (task == OptimTask.ERROR) {
                     fatal(solver.getReason());
                 }
                 if (task == OptimTask.WARNING) {
                     warn(solver.getReason());
                     break;
                 }
                 if (job.verbose && (task == OptimTask.NEW_X || task == OptimTask.FINAL_X)) {
                     double elapsed = solver.getElapsedTime();
                     int evaluations = solver.getEvaluations();
                     int iterations = solver.getIterations();
                     solver.getRestarts();
                     job.stream.format("iter: %4d    eval: %4d    time: %7.3f s.    fx = %22.16e    |gx| = %8.2e\n",
                             iterations, evaluations,
                             elapsed, solver.getCost(),
                             solver.getGradient().norm2());
                     if (task == OptimTask.FINAL_X) {
                         job.stream.format("Total time in cost function: %.3f s (%.3f ms/eval.)\n",
                                 elapsed, (evaluations > 0 ? 1e3*elapsed/evaluations : 0.0));
                     }
                     // if (fdata instanceof WeightedConvolutionCost) {
                     //     WeightedConvolutionCost f = fdata;
                     //     elapsed = f.getElapsedTimeInFFT();
                     //     System.out.format("Total time in FFT: %.3f s (%.3f ms/eval.)\n",
                     //         elapsed, (evaluations > 0 ? 1e3*elapsed/evaluations : 0.0));
                     //     elapsed = f.getElapsedTime() - elapsed;
                     //     System.out.format("Total time in other parts of the convolution operator: %.3f s (%.3f ms/eval.)\n",
                     //         elapsed, (evaluations > 0 ? 1e3*elapsed/evaluations : 0.0));
                     // }
                 }
                 if (task == OptimTask.FINAL_X) {
                     break;
                 }
                 task = solver.iterate();
             }
        } catch (RuntimeException e) {
            fatal(e.getMessage());
        }
        try {
            ShapedArray arr = solver.getBestSolution().asShapedArray();
            //arr = ArrayUtils.roll(arr);
            if (job.crop) {
                arr = ArrayUtils.crop(arr, solver.getData().getShape());
            }
            MainCommand.saveArrayToOMETiff(outputName, arr);
        } catch (final IOException e) {
            if (job.debug) {
                e.printStackTrace();
            }
            fatal("Failed to write output image (" + e.getMessage() + ")");
        }
        System.exit(0);
    }

    private final static boolean isnan(double x) {
        return Double.isNaN(x);
    }

    private static void fatal(String mesg) {
        System.err.format("Error: %s.\n", mesg);
        System.exit(1);
    }

    private static void warn(String mesg) {
        System.err.format("Warning: %s.\n", mesg);
    }
}
