package commands;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.Arrays;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.mitiv.TiPi.array.Array3D;
import org.mitiv.TiPi.array.ArrayFactory;
import org.mitiv.TiPi.array.ShapedArray;
import org.mitiv.TiPi.base.Shape;
import org.mitiv.TiPi.base.Traits;
import org.mitiv.TiPi.io.ColorModel;
import org.mitiv.TiPi.utils.FFTUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.common.services.ServiceFactory;
import loci.formats.FormatException;
import loci.formats.FormatTools;
import loci.formats.IFormatWriter;
import loci.formats.ImageReader;
import loci.formats.ImageWriter;
import loci.formats.meta.IMetadata;
import loci.formats.services.OMEXMLService;
import ome.xml.model.enums.DimensionOrder;
import ome.xml.model.enums.PixelType;
import ome.xml.model.primitives.PositiveInteger;

public class MainCommand {
    private PrintStream stream = System.out;

    @Option(name="prog", usage="choose the program to use: deconv or blinddeconv")
    private String arg1;

    @Argument
    private String args;

    final static Logger logger = LoggerFactory.getLogger(MainCommand.class);

    public static void main(String[] args) throws FormatException, IOException, DependencyException, ServiceException {
        MainCommand job = new MainCommand();
        ch.qos.logback.classic.Logger root = (ch.qos.logback.classic.Logger) LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(ch.qos.logback.classic.Level.ERROR);
        if (args.length > 1){
            String[] newArgs = Arrays.copyOfRange(args, 1, args.length);
            if (args[0].equals("deconv")) {
                EdgePreservingDeconvolutionCommand.main(newArgs);
                return;
            } else if (args[0].equals("blinddeconv")) {
                BlindDeconvolutionCommand.main(newArgs);
                return;
            }
        }
        CmdLineParser parser = new CmdLineParser(job);
        job.stream.println("Usage: microtipi prog [OPTIONS] INPUT OUTPUT");
        parser.printUsage(job.stream);
    }

    public static ShapedArray readOMETiffToArray(String path) throws FormatException, IOException {
        ImageReader reader = new ImageReader();
        logger.info("Start reading OME-Tiff: {}", path);
        reader.setId(path);
        if (reader.getSeriesCount()>1 || reader.getSizeT()>1 || reader.getSizeC()>1) {
            int s = reader.getSeriesCount();
            int t = reader.getSizeT();
            int c = reader.getSizeC();
            int z = reader.getSizeZ();
            reader.close();
            throw new FormatException(String.format("File no good shape (Series:%d, T:%d, C:%d, Z:%d)", s, t, c, z));
        }
        reader.setSeries(0);
        int bytesPerPixel = reader.getBitsPerPixel() / 8;
        int sizeX = reader.getSizeX();
        int sizeY = reader.getSizeY();
        int sizeZ = reader.getSizeZ();
        logger.info("reader size ZYX: {}x{}x{}", sizeZ, sizeY, sizeX);
        logger.info("reader bits per pixel: {}", bytesPerPixel);
        // Calculate the size in bits
        int bufferSizeInBytes = bytesPerPixel * sizeX * sizeY * sizeZ;
        logger.info("buffer size in bytes: {}", bufferSizeInBytes);
    
        ShapedArray shapedArray = null;
        ByteBuffer byteBuffer = ByteBuffer.allocate(bufferSizeInBytes);
        logger.info("Converting OME-Tiff to Array");
        logger.debug("reader size Z: {}", reader.getSizeZ());
        for (int i=0; i<reader.getSizeZ(); i++) {
            try {
                logger.trace("openBytes {}", i);
                byte[] plane = reader.openBytes(i);
                logger.trace("ByteBuffer.put: plane of size {}", plane.length);
                logger.trace("ByteBuffer remaining : {}", byteBuffer.remaining());
                byteBuffer.put(plane);
            } catch (Exception e) {
                logger.error("Failed to read plane at index {}: {}", i, e.getMessage());
                e.printStackTrace();
            }
        
        }
        if (reader.isLittleEndian()){
            byteBuffer.order(ByteOrder.LITTLE_ENDIAN);
        }
        logger.debug("reader pixel type is {}", reader.getPixelType());
        switch (reader.getPixelType()) {
            case FormatTools.INT8:
                byteBuffer.flip();
                IntBuffer unsignedByteBuffer = IntBuffer.allocate(byteBuffer.capacity());
                byte a = (byte) 0b10000000;
                while (byteBuffer.hasRemaining()) {
                    byte signedValue = byteBuffer.get();
                    byte unsignedValue;
                    unsignedValue = (byte) (((a & signedValue) ^ a) & a | (~a & signedValue));
                    unsignedByteBuffer.put(unsignedValue);
                }
                shapedArray = ArrayFactory.wrap(unsignedByteBuffer.array(), reader.getSizeX(), reader.getSizeY(), reader.getSizeZ());
                break;
            case FormatTools.UINT8:
                shapedArray = ArrayFactory.wrap(byteBuffer.array(), reader.getSizeX(), reader.getSizeY(), reader.getSizeZ());
                break;
            case FormatTools.INT16:
                ShortBuffer shortBuffer = ShortBuffer.allocate(byteBuffer.capacity()/2);
                byteBuffer.flip();
                while (byteBuffer.hasRemaining()) {
                    byte low = byteBuffer.get();
                    byte high = byteBuffer.get();
                    short s = (short) ((high << 8) | low);
                    shortBuffer.put(s);
                }

                ShortBuffer unsignedShortBuffer = ShortBuffer.allocate(byteBuffer.capacity()/2);
                short aShort = (short) 0b1000000000000000;
                shortBuffer.flip();
                while (shortBuffer.hasRemaining()) {
                    short signedValue = shortBuffer.get();
                    short unsignedValue;
                    unsignedValue = (short) ~(((aShort & signedValue) ^ aShort) & aShort | (~aShort & signedValue));
                    unsignedShortBuffer.put(unsignedValue);
                }
                
                shapedArray = ArrayFactory.wrap(unsignedShortBuffer.array(), reader.getSizeX(), reader.getSizeY(), reader.getSizeZ());
                break;
            case FormatTools.UINT16:
                ShortBuffer shortBuffer2 = ShortBuffer.allocate(byteBuffer.capacity()/2);
                byteBuffer.flip();
                while (byteBuffer.hasRemaining()) {
                    byte low = byteBuffer.get();
                    byte high = byteBuffer.get();
                    short s = (short) ((high << 8) | low);
                    shortBuffer2.put(s);
                }
                shapedArray = ArrayFactory.wrap(shortBuffer2.array(), reader.getSizeX(), reader.getSizeY(), reader.getSizeZ());
                break;
            case FormatTools.FLOAT:
                FloatBuffer floatBuffer = FloatBuffer.allocate(byteBuffer.capacity()/4);
                byteBuffer.flip();
                reader.isLittleEndian();
                while (byteBuffer.hasRemaining()) {
                    float s = byteBuffer.getFloat();
                    floatBuffer.put(s);
                }
                shapedArray = ArrayFactory.wrap(floatBuffer.array(), reader.getSizeX(), reader.getSizeY(), reader.getSizeZ());
                shapedArray = shapedArray.toFloat();
                break;
            case FormatTools.DOUBLE:
                DoubleBuffer doubleBuffer = DoubleBuffer.allocate(byteBuffer.capacity()/8);
                byteBuffer.flip();
                reader.isLittleEndian();
                while (byteBuffer.hasRemaining()) {
                    double s = byteBuffer.getDouble();
                    doubleBuffer.put(s);
                }
                shapedArray = ArrayFactory.wrap(doubleBuffer.array(), reader.getSizeX(), reader.getSizeY(), reader.getSizeZ());
                shapedArray = shapedArray.toDouble();
                break;
            default:
                reader.close();
                throw new IOException("format not supported", null);
        }
        reader.close();
        logger.info("Reader closed");
        return shapedArray;
    }

    public static void saveArrayToOMETiff(String path, ShapedArray arr)
    throws DependencyException, ServiceException, FormatException, IOException {
        ServiceFactory factory = new ServiceFactory();
        OMEXMLService service = factory.getInstance(OMEXMLService.class);
        IMetadata omexml = service.createOMEXMLMetadata();
        omexml.setImageID("Image:0", 0);
        omexml.setPixelsID("Pixels:0", 0);
        omexml.setPixelsBinDataBigEndian(Boolean.FALSE, 0, 0);
        omexml.setPixelsDimensionOrder(DimensionOrder.XYCZT, 0);
        switch (arr.getType()) {
            case Traits.BYTE:
                omexml.setPixelsType(PixelType.INT8, 0);
                break;
            case Traits.SHORT:
                omexml.setPixelsType(PixelType.INT16, 0);
                break;
            case Traits.INT:
                omexml.setPixelsType(PixelType.INT32, 0);
                break;
            case Traits.FLOAT:
                omexml.setPixelsType(PixelType.FLOAT, 0);
                break;
            case Traits.DOUBLE:
                omexml.setPixelsType(PixelType.DOUBLE, 0);
                break;
            case Traits.BOOLEAN:
                omexml.setPixelsType(PixelType.BIT, 0);
                break;
            default:
                String message = "arr type should be short, int, long, float, double, or boolean found " + arr.getType();
                throw new IOException(message, null);
        }
        if (arr.getRank() != 3) {
            throw new IOException("arr rank should be 3", null);
        }
        omexml.setPixelsSizeX(new PositiveInteger(arr.getDimension(0)), 0);
        omexml.setPixelsSizeY(new PositiveInteger(arr.getDimension(1)), 0);
        omexml.setPixelsSizeZ(new PositiveInteger(arr.getDimension(2)), 0);
        omexml.setPixelsSizeT(new PositiveInteger(1), 0);
        omexml.setPixelsSizeC(new PositiveInteger(1), 0);
        omexml.setChannelID("Channel:0:0", 0, 0);
        omexml.setChannelSamplesPerPixel(new PositiveInteger(1),0, 0);
    
        ImageWriter imwriter = new ImageWriter();
        imwriter.setMetadataRetrieve(omexml);
        File file = new File(path); 
        if (!file.exists() || file.delete()) {
            imwriter.setId(path);
            IFormatWriter writer = imwriter.getWriter();
            Array3D data = (Array3D) arr;
            for (int image=0; image<arr.getDimension(2); image++) {
                ByteBuffer bb = null;
                switch (arr.getType()) {
                    case Traits.BYTE:
                        byte[] bytePlane = (byte[]) data.slice(image, 2).flatten(true);
                        bb = ByteBuffer.allocate(bytePlane.length);
                        for (byte d: bytePlane) {
                            bb.put(d);
                        }
                        break;
                    case Traits.SHORT:
                        short[] shortPlane = (short[]) data.slice(image, 2).flatten(true);
                        bb = ByteBuffer.allocate(shortPlane.length * 2);
                        for (short d: shortPlane) {
                            bb.putShort(d);
                        }
                        break;
                    case Traits.INT:
                        int[] intPlane = (int[]) data.slice(image, 2).flatten(true);
                        bb = ByteBuffer.allocate(intPlane.length * 4);
                        for (int d: intPlane) {
                            bb.putInt(d);
                        }
                        break;
                    case Traits.LONG:
                        long[] longPlane = (long[]) data.slice(image, 2).flatten(true);
                        bb = ByteBuffer.allocate(longPlane.length * 8);
                        for (long d: longPlane) {
                            bb.putLong(d);
                        }
                        break;
                    case Traits.FLOAT:
                        float[] floatPlane = (float[]) data.slice(image, 2).flatten(true);
                        bb = ByteBuffer.allocate(floatPlane.length * 4);
                        for (float d: floatPlane) {
                            bb.putFloat(d);
                        }
                        break;
                    case Traits.DOUBLE:
                        double[] doublePlane = (double[]) data.slice(image, 2).flatten(true);
                        bb = ByteBuffer.allocate(doublePlane.length * 8);
                        for (double d: doublePlane) {
                            bb.putDouble(d);
                        }
                        break;
                    case Traits.BOOLEAN:
                        throw new IOException("Boolean are not implemented", null);
                }
                if (bb != null) {
                    writer.saveBytes(image, bb.array());
                } else {
                    throw new IOException("type problem", null);
                }
            }
            writer.close();
            imwriter.close();
        }
    
    }

    public static ShapedArray loadData(String name, boolean single) throws FormatException, IOException {
        ShapedArray arr = readOMETiffToArray(name);
        ColorModel colorModel = ColorModel.guessColorModel(arr);
        if (colorModel == ColorModel.NONE) {
            return (single ? arr.toFloat() :  arr.toDouble());
        } else {
            return (single
                    ? ColorModel.filterImageAsFloat(arr, ColorModel.GRAY)
                            : ColorModel.filterImageAsDouble(arr, ColorModel.GRAY));
        }
    }

    static int[] getPaddingShape(String paddingMethod, Shape dataShape) {
        Shape nullShape = new Shape(1, 1, 1);
        return getPaddingShape(paddingMethod, dataShape, nullShape);
    }

    static int[] getPaddingShape(String paddingMethod, Shape dataShape, Shape psfShape) {
        int rank = dataShape.rank();
        int[] objDims = new int[rank];
        if (paddingMethod.equals("auto")) {
            for (int k = 0; k < rank; ++k) {
                int dataDim = dataShape.dimension(k);
                int psfDim = psfShape.dimension(k);
                objDims[k] = FFTUtils.bestDimension(dataDim + psfDim - 1);
            }
        } else if (paddingMethod.equals("min")) {
            for (int k = 0; k < rank; ++k) {
                int dataDim = dataShape.dimension(k);
                int psfDim = psfShape.dimension(k);
                objDims[k] = FFTUtils.bestDimension(Math.max(dataDim, psfDim));
            }
        } else {
            int pad;
            try {
                pad = Integer.parseInt(paddingMethod);
            } catch (NumberFormatException ex) {
                throw new IllegalArgumentException("Invalid value for option `-pad`, must be \"auto\", \"min\" or an integer");
            }
            if (pad < 0) {
                throw new IllegalArgumentException("Padding value must be nonnegative");
            }
            for (int k = 0; k < rank; ++k) {
                int dataDim = dataShape.dimension(k);
                int psfDim = psfShape.dimension(k);
                objDims[k] = FFTUtils.bestDimension(Math.max(dataDim, psfDim) + pad);
            }
        }
        return objDims;
    }
}
