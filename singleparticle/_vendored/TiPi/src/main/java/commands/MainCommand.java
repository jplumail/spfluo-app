package commands;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.util.Arrays;

import org.kohsuke.args4j.Argument;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;
import org.mitiv.TiPi.array.Array3D;
import org.mitiv.TiPi.array.ArrayFactory;
import org.mitiv.TiPi.array.ShapedArray;
import org.mitiv.TiPi.base.Traits;
import org.mitiv.TiPi.io.ColorModel;

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

    public static void main(String[] args) throws FormatException, IOException, DependencyException, ServiceException {
        MainCommand job = new MainCommand();
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

    static ShapedArray readOMETiffToArray(String path) throws FormatException, IOException {
        ImageReader reader = new ImageReader();
        reader.setId(path);
        if (reader.getSeriesCount()>1 || reader.getSizeT()>1 || reader.getSizeC()>1) {
            reader.close();
            throw new FormatException("File no good shape (Series:%d, T:%d, C:%d)".formatted(reader.getSeriesCount(), reader.getSizeT(), reader.getSizeC()));
        }
        reader.setSeries(0);
        int bitsPerPixel = reader.getBitsPerPixel();
        int sizeX = reader.getSizeX();
        int sizeY = reader.getSizeY();
        int sizeZ = reader.getSizeZ();
        // Calculate the size in bits
        int bufferSizeInBits = bitsPerPixel * sizeX * sizeY * sizeZ;
    
        ShapedArray shapedArray = null;
        ByteBuffer buffer = ByteBuffer.allocate(bufferSizeInBits / 8);
        for (int i=0; i<reader.getSizeZ(); i++) {
            byte[] plane = reader.openBytes(i);
            buffer.put(plane);
        }
        shapedArray = ArrayFactory.wrap(buffer.array(), reader.getSizeX(), reader.getSizeY(), reader.getSizeZ());
        switch (reader.getPixelType()) {
            case FormatTools.INT8:
                reader.close();
                throw new IOException("INT8 format not supported", null);
            case FormatTools.UINT8:
                shapedArray = shapedArray.toByte(); // TiPi Byte corresponds to unsigned int
                break;
            case FormatTools.INT16:
                shapedArray = shapedArray.toShort();
                reader.close();
                throw new IOException("INT16 format not supported", null);
            case FormatTools.UINT16:
                reader.close();
                throw new IOException("UINT16 format not supported", null);
            case FormatTools.INT32:
                shapedArray = shapedArray.toInt();
                reader.close();
                throw new IOException("INT32 format not supported", null);
            case FormatTools.UINT32:
                reader.close();
                throw new IOException("UINT32 format not supported", null);
            case FormatTools.FLOAT:
                shapedArray = shapedArray.toFloat();
                reader.close();
                throw new IOException("FLOAT format not supported", null);
            case FormatTools.DOUBLE:
                shapedArray = shapedArray.toDouble();
                reader.close();
                throw new IOException("DOUBLE format not supported", null);
            case FormatTools.BIT:
                reader.close();
                throw new IOException("BIT format not supported", null);
        }
        reader.close();
        return shapedArray;
    }

    static void saveArrayToOMETiff(String path, ShapedArray arr)
    throws DependencyException, ServiceException, FormatException, IOException {
        ServiceFactory factory = new ServiceFactory();
        OMEXMLService service = factory.getInstance(OMEXMLService.class);
        IMetadata omexml = service.createOMEXMLMetadata();
        omexml.setImageID("Image:0", 0);
        omexml.setPixelsID("Pixels:0", 0);
        omexml.setPixelsBinDataBigEndian(Boolean.TRUE, 0, 0);
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
}
