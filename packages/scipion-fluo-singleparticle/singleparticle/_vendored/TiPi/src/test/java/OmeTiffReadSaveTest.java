

import static commands.MainCommand.readOMETiffToArray;
import static commands.MainCommand.saveArrayToOMETiff;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mitiv.TiPi.array.ShapedArray;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import static org.junit.jupiter.api.Assertions.assertTrue;

public class OmeTiffReadSaveTest {

    private static final String TEST_DIRECTORY = System.getProperty("user.dir") + "/src/test/data";
    private static Path tempDirectory;

    @BeforeAll
    public static void setup() throws Exception {
        // Créer le répertoire temporaire pour les images sauvegardées
        //tempDirectory = Files.createTempDirectory("testTempDir");
        tempDirectory = Paths.get(System.getProperty("user.dir") + "/src/test/tmp/");
        Files.createDirectories(tempDirectory);
    }

    // @AfterAll
    // public static void tearDown() throws Exception {
    //     // Nettoyer le répertoire temporaire après les tests
    //     Files.walk(tempDirectory)
    //         .sorted(Comparator.reverseOrder())
    //         .map(Path::toFile)
    //         .forEach(File::delete);
    // }

    static byte[] removeUUIDs(byte[] inputBytes) {
        String text = new String(inputBytes, StandardCharsets.UTF_8);

        // Regex pour matcher les UUIDs de la forme spécifiée
        String uuidRegex = "urn:uuid:[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}";
        Pattern pattern = Pattern.compile(uuidRegex);
        Matcher matcher = pattern.matcher(text);

        // Suppression des UUIDs
        String modifiedText = matcher.replaceAll("");

        return modifiedText.getBytes(StandardCharsets.UTF_8);
    }


    @ParameterizedTest
    @ValueSource(strings = {
            "test-double.ome.tiff",
            "test-float.ome.tiff",
            //"test-int8.ome.tiff",
            //"test-int16.ome.tiff",
            //"test-int32.ome.tiff",
    })
    public void testOMETiffReadWrite(String fileName) throws Exception {
        String originalPath = TEST_DIRECTORY + "/" + fileName;
        String savedPath = tempDirectory + "/" + fileName;

        // Lire l'image originale
        ShapedArray originalArray = readOMETiffToArray(originalPath);

        // Sauvegarder l'image lue
        saveArrayToOMETiff(savedPath, originalArray);

        // Comparer
        ShapedArray savedArray = readOMETiffToArray(savedPath);
        
        assertTrue(java.util.Arrays.equals(originalArray.toByte().flatten(), savedArray.toByte().flatten()),
                "Les array doivent être identiques");

    }
}
