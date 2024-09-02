package commands;

import java.io.IOException;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.mitiv.microTiPi.epifluorescence.WideFieldModel;
import org.mitiv.microTiPi.microUtils.BlindDeconvJob;
import org.mitiv.microTiPi.microscopy.PSF_Estimation;
import org.mitiv.TiPi.array.ArrayFactory;
import org.mitiv.TiPi.array.ArrayUtils;
import org.mitiv.TiPi.array.ByteArray;
import org.mitiv.TiPi.array.DoubleArray;
import org.mitiv.TiPi.array.FloatArray;
import org.mitiv.TiPi.array.ShapedArray;
import org.mitiv.TiPi.base.Shape;
import org.mitiv.TiPi.base.Traits;
import org.mitiv.TiPi.conv.WeightedConvolutionCost;
import org.mitiv.TiPi.cost.DifferentiableCostFunction;
import org.mitiv.TiPi.cost.HyperbolicTotalVariation;
import org.mitiv.TiPi.jobs.DeconvolutionJob;
import org.mitiv.TiPi.linalg.shaped.DoubleShapedVectorSpace;
import org.mitiv.TiPi.linalg.shaped.FloatShapedVectorSpace;
import org.mitiv.TiPi.linalg.shaped.ShapedVectorSpace;
import org.mitiv.TiPi.utils.HistoMap;
import org.mitiv.io.DeconvHook;
import org.mitiv.io.NullImager;
import org.mitiv.TiPi.weights.WeightFactory;
import org.mitiv.TiPi.weights.WeightUpdater;
import org.mitiv.TiPi.weights.weightsFromModel;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.formats.FormatException;

public class BlindDeconvolutionCommand {

    @Option(name = "-nbloops", usage = "number of loops", metaVar = "N")
    private int loops = 1;

    @Option(name = "-maxIterDefocus", usage = "Max number of iterations for defocus", metaVar = "N")
    private int maxIterDefocus = 20;

    @Option(name = "-maxIterPhase", usage = "Max number of iterations for phase", metaVar = "N")
    private int maxIterPhase = 20;

    @Option(name = "-maxIterModulus", usage = "Max number of iterations for modulus", metaVar = "N")
    private int maxIterModulus = 0;

    // Padding
    @Option(name = "-pad", usage = "Padding method.", metaVar = "\"auto\"|\"min\"|NUMBER")
    private String paddingMethod = "30";

    // WidefieldModel args
    @Option(name = "-nPhase", usage = "Number of zernike describing the pupil phase", metaVar = "N")
    private int nPhase = 19;

    @Option(name = "-nModulus", usage = "Number of zernike describing the pupil modulus", metaVar = "N")
    private int nModulus = 0;

    @Option(name = "-NA", usage = "Numerical aperture", metaVar = "NA")
    private double NA = 1.4;

    @Option(name = "-lambda", usage = "Wavelength in nm", metaVar = "lambda")
    private double lambda = 540; // 540nm

    @Option(name = "-ni", usage = "Refractive index", metaVar = "ni")
    private double ni = 1.518;

    @Option(name = "-dxy", usage = "Lateral pixel size in nm", metaVar = "dxy")
    private double dxy = 1.;

    @Option(name = "-dz", usage = "Axial pixel size in nm", metaVar = "dz")
    private double dz = 1.;

    @Option(name = "-radial", usage = "Radial option")
    private boolean radial;

    @Option(name = "-single", usage = "Compute in single precision")
    private boolean single;

    // Noise model args
    private enum WeightingMethod { CONSTANT, VAR_MAP, INVERSE_VAR_MAP, COMPUTED_VAR_MAP, AUTOMATIC_VAR_MAP };
    @Option(name = "-weighting", metaVar = "weighting")
    private WeightingMethod weighting = WeightingMethod.COMPUTED_VAR_MAP;

    // if weighting is VAR_MAP or INVERSE_VAR_MAP
    @Option(name = "-map", usage = "Map of space varying noise or precision", metaVar = "mapPath")
    private String mapPath;

    // if weighting is COMPUTED_VAR_MAP or AUTOMATIC_VAR_MAP
    @Option(name = "-gain", usage = "The detector gain in electrons per analog digital unit (ADU)", metaVar = "gain")
    private double gain = Double.NaN;

    @Option(name = "-readoutNoise", usage = "The standard deviation of the read-out noise in electron/pixel", metaVar = "sigma")
    private double sigma = Double.NaN;

    // optional baddata map
    @Option(name = "-badDataMap", usage = "A binary map with 0 for bad pixels", metaVar = "badDataPath")
    private String badDataPath;

    // Deconvolution args
    @Option(name = "-epsilon", usage = "threshold of Hyperbolic TV", metaVar = "epsilon")
    private double epsilon = 0.01;

    @Option(name = "-mu", usage = "Value of mu. Must be positive", metaVar = "mu")
    private double mu = 1.;

    @Option(name = "-negativity", usage = "Allow negativity")
    private boolean negativity;

    @Option(name = "-nbIterDeconv", usage = "Number of iterations for deconvolution", metaVar = "N")
    private int nbIterDeconv = 50;

    @Option(name = "-crop", usage = "Crop result to same size as input.")
    private boolean crop = false;

    // Misc
    @Option(name = "-debug", usage = "debug flag")
    private boolean debug;

    @Option(name = "-help", aliases = {"--help", "-h", "-?"}, usage = "Display help.")
    private boolean help;

    @Argument
    private List<String> arguments;

    static private void usage(CmdLineParser parser, int code) {
        PrintStream stream = (code == 0 ? System.out : System.err);
        stream.println("Usage: blinddeconv [OPTIONS] INPUT OUTPUT OUTPUT_PSF");
        if (code == 0) {
            stream.println("Options:");
            parser.getProperties().withUsageWidth(80);
            parser.printUsage(stream);
        } else {
            stream.println("Try option -help for a more complete description of options.");
        }
        System.exit(code);
    }

    private static WideFieldModel buildPupil(BlindDeconvolutionCommand job, Shape psfShape) {
        WideFieldModel pupil = new WideFieldModel(psfShape, job.nPhase, job.nModulus, job.NA, job.lambda*1E-9, job.ni, job.dxy*1E-9, job.dz*1E-9, job.radial, job.single);
        pupil.setPupilAxis(new double[]{0., 0.});
        pupil.setModulus(new double[]{1.});
        return pupil;
    }

    protected static ShapedArray getWeights(BlindDeconvolutionCommand job, ShapedArray dataArray, boolean normalize) throws IllegalArgumentException, FormatException, IOException {

        ShapedArray wgtArray = null;
        Shape dataShape = dataArray.getShape();
        ByteArray badpixArray = (ByteArray) ArrayFactory.create(Traits.BYTE, dataShape);
        if (job.badDataPath != null) {
            throw new IllegalArgumentException("custom bad data is not yet implemented");
        } 
        if (job.single) {
            WeightFactory.flagBads(dataArray, badpixArray, ((FloatArray) dataArray).max());
        } else {
            WeightFactory.flagBads(dataArray, badpixArray, ((DoubleArray) dataArray).max());
        }
        
        if (job.single){
            float maxi = dataArray.toFloat().max();
            WeightFactory.flagBads(dataArray, badpixArray, maxi);
        } else {
            double maxi = dataArray.toDouble().max();
            WeightFactory.flagBads(dataArray, badpixArray, maxi);
        }

        if (job.weighting == WeightingMethod.INVERSE_VAR_MAP) {
            // A map of weights is provided.
            if (job.mapPath != null) {
                wgtArray = MainCommand.loadData(job.mapPath, job.single);
                if (!wgtArray.getShape().equals(dataShape)) {
                    throw new IllegalArgumentException("Weight map must have the same size than the data");
                }
            }
        } else if (job.weighting == WeightingMethod.VAR_MAP) {
            // A variance map is provided. FIXME: check shape and values. (ferreol comment)
            if (job.mapPath != null) {
                ShapedArray varArray = MainCommand.loadData(job.mapPath, job.single);
                if (!varArray.getShape().equals(dataShape)) {
                    throw new IllegalArgumentException("Weight map must have the same size than the data");
                }
                wgtArray = WeightFactory.computeWeightsFromVariance(varArray);
            }
        } else if (job.weighting == WeightingMethod.COMPUTED_VAR_MAP) {
            // Weights are computed given the gain and the readout noise of the detector.
            double alpha;
            double beta;
            if (Double.isNaN(job.sigma)) {
                if (! Double.isNaN(job.gain)) {
                    System.err.println("Warning: linear noise model parameter is ignored if affine noise model parameter is not specified");
                }
                alpha = 0;
                beta = 1;
            } else if (Double.isNaN(job.gain)) {
                alpha = 0;
                beta = job.sigma * job.sigma;
            } else {
                alpha = 1/job.gain;
                beta = (job.sigma/job.gain) * (job.sigma/job.gain);
            }
            wgtArray = WeightFactory.computeWeightsFromData(dataArray, alpha, beta);
        } else if (job.weighting == WeightingMethod.AUTOMATIC_VAR_MAP) {
            // Weights are computed from the current estimate of the data modelArray =object * PSF
            // the gain and the readout noise of the detector are automatically estimated from the variance of the data given the modelArray
            ShapedArray modelArray = MainCommand.loadData(job.mapPath, job.single);
            HistoMap hm = new HistoMap(modelArray, dataArray, badpixArray);
            wgtArray = hm.computeWeightMap(modelArray);
            throw new IllegalArgumentException("automatic_var_map is not yet implemented.");
        } else {
            // variance = 1
            if (job.single) {
                wgtArray = ArrayFactory.create(Traits.FLOAT, dataShape);
                ((FloatArray) wgtArray).fill(1.0f);
            }else{
                wgtArray = ArrayFactory.create(Traits.DOUBLE, dataShape);
                ((DoubleArray) wgtArray).fill(1.0);
            }
        }

        // WeightFactory.removeBads(wgtArray, badpixArray); FIXME: not implemented

        if (normalize) {
            WeightFactory.normalize(wgtArray);
        }

        return wgtArray;
    }


    public static void main(String[] args) throws FormatException, IOException, DependencyException, ServiceException {

        // Switch to "US" locale to avoid problems with number formats.
        Locale.setDefault(Locale.US);

        // Parse options.
        BlindDeconvolutionCommand job = new BlindDeconvolutionCommand();
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
        if (size != 3) {
            System.err.format("Too %s arguments.\n", (size < 2 ? "few" : "many"));
            usage(parser, 1);
        }
        String inputName = job.arguments.get(0);
        String outputName = job.arguments.get(1);
        String psfName = job.arguments.get(2);
        
        // Read input file
        ShapedArray dataArray = MainCommand.loadData(inputName, job.single);

        // Compute output shape
        ShapedVectorSpace dataSpace, objectSpace;
        int[] shape = MainCommand.getPaddingShape(job.paddingMethod, dataArray.getShape());
        int Nxy = Math.max(shape[0], shape[1]);
        int Nz = shape[2];
        Shape psfShape = new Shape(Nxy, Nxy, Nz);
        Shape outputShape = new Shape(Nxy, Nxy, Nz);

        // build objArray, the deconvolved image
        ShapedArray objArray = dataArray.copy();
        objArray = ArrayUtils.extract(objArray, outputShape, 0.); //Padding to the right size

        // build pupil, psf
        WideFieldModel pupil = buildPupil(job, psfShape);
        PSF_Estimation psfEstimation = new PSF_Estimation(pupil);
        ShapedArray psfArray = ArrayUtils.roll( pupil.getPsf() );

        // build hooks
        NullImager imager = new NullImager();
        DeconvHook dHook = new DeconvHook(imager, outputShape,null, job.debug);
        DeconvHook dHookfinal = new DeconvHook(imager, outputShape,"Deconvolved", job.debug);

        // build wgtArray, weights
        ShapedArray wgtArray = getWeights(job, dataArray, true);
        WeightUpdater wghtUpdt = null;
        if (job.weighting == WeightingMethod.AUTOMATIC_VAR_MAP) {
            wghtUpdt = new weightsFromModel( dataArray, null);
        }

        // build deconvolver
        if (job.single) {
            dataSpace = new FloatShapedVectorSpace(dataArray.getShape());
            objectSpace = new FloatShapedVectorSpace(outputShape);
        }
        else {
            dataSpace = new DoubleShapedVectorSpace(dataArray.getShape());
            objectSpace = new DoubleShapedVectorSpace(outputShape);
        }
        double[] scale = {1, 1, job.dz / job.dxy};
        DifferentiableCostFunction fprior = new HyperbolicTotalVariation(objectSpace, job.epsilon, scale);
        WeightedConvolutionCost fdata =  WeightedConvolutionCost.build(objectSpace, dataSpace);
        fdata.setData(dataArray);
        fdata.setPSF(psfArray);
        fdata.setWeights(wgtArray,true);
        psfEstimation.setWeight(ArrayUtils.pad(wgtArray,psfShape));
        DeconvolutionJob deconvolver = new DeconvolutionJob(fdata, job.mu, fprior, !job.negativity, job.nbIterDeconv, dHook, dHookfinal); // hooks to null

        // update psfEstimation
        psfEstimation.setData(objArray);
        psfEstimation.enablePositivity(false);
        psfEstimation.setAbsoluteTolerance(0.0);

        // build bdec
        BlindDeconvJob bdec = new BlindDeconvJob(
            job.loops,
            pupil.getParametersFlags(),
            new int[] {job.maxIterDefocus, job.maxIterPhase, job.maxIterModulus},
            psfEstimation,
            deconvolver,
            wghtUpdt,
            job.debug
        );

        // run blinddeconv
        // System.out.println("Blind deconv");
        // long startTime = System.nanoTime();
        objArray = bdec.blindDeconv(objArray);
        // long endTime = System.nanoTime();
        // long elapsedTime = endTime - startTime;
        // double elapsedTimeInSeconds = (double) elapsedTime / 1_000_000_000.0;
        // System.out.println("Elapsed Time: " + elapsedTimeInSeconds + " seconds");

        if (wghtUpdt != null) {
            System.out.print("Gain found: ");
            System.out.println(((weightsFromModel) wghtUpdt).getAlpha());
            System.out.print("Noise found: ");
            System.out.println(Math.sqrt(((weightsFromModel) wghtUpdt).getBeta())/((weightsFromModel) wghtUpdt).getAlpha());
        }
        if(job.maxIterDefocus>0){
            System.out.print("Ni found: ");
            System.out.println(((WideFieldModel) psfEstimation.getModel()).getNi());
            System.out.print("Pupil shift found: ");
            System.out.println(Arrays.toString(((WideFieldModel) psfEstimation.getModel()).getPupilShift()));
        }
        if(job.maxIterPhase>0){
            System.out.print("Phase coefs found: ");
            System.out.println(Arrays.toString(((WideFieldModel) psfEstimation.getModel()).getPhaseCoefs().getData()));
        }
        if(job.maxIterModulus>0){
            System.out.print("Mudulus coefs found: ");
            System.out.println(Arrays.toString(((WideFieldModel) psfEstimation.getModel()).getModulusCoefs().getData()));
        }
        
        // get the psf
        pupil = ((WideFieldModel) psfEstimation.getModel());
        pupil.freeMem();

        // save arrays
        psfArray = pupil.getPsf();
        psfArray = ArrayUtils.roll(psfArray);
        if (job.crop) {
            objArray = ArrayUtils.crop(objArray, dataArray.getShape());
            psfArray = ArrayUtils.crop(psfArray, dataArray.getShape());
        }
        MainCommand.saveArrayToOMETiff(outputName, objArray);
        MainCommand.saveArrayToOMETiff(psfName, psfArray);
    }
}
