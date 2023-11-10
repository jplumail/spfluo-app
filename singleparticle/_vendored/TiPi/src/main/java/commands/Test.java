package commands;

import java.io.IOException;

import org.mitiv.TiPi.array.ShapedArray;

import loci.common.services.DependencyException;
import loci.common.services.ServiceException;
import loci.formats.FormatException;


public class Test {
    public static void main(String[] args) throws DependencyException, ServiceException, FormatException, IOException {
        System.out.println("byte=" + (byte) 0x80);
        String path = "in-uint8.ome.tiff";
        String path_out = "out-uint8.ome.tiff";
        ShapedArray arr = MainCommand.readOMETiffToArray(path);
        MainCommand.saveArrayToOMETiff(path_out, arr);
    }
}
