import org.apache.batik.transcoder.TranscoderInput;
import org.apache.batik.transcoder.TranscoderOutput;
import org.apache.batik.transcoder.image.PNGTranscoder;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.File;

public class OnloadTest {
    public static void main(String[] args) {
        // Specify the input folder containing SVG files and the output folder for PNG files
        String svgFolderPath = "/Users/wang/Documents/project/configuration_code_understanding/code4/data/onload_test/";
        String pngOutputFolderPath = "/Users/wang/Documents/project/configuration_code_understanding/code4/data/batik/output/";

        File svgFolder = new File(svgFolderPath);
        File[] svgFiles = svgFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".svg"));

        if (svgFiles == null || svgFiles.length == 0) {
            System.out.println("No SVG files found!");
            return;
        }

        // Create output folder if it does not exist
        File pngOutputFolder = new File(pngOutputFolderPath);
        if (!pngOutputFolder.exists()) {
            pngOutputFolder.mkdirs();
        }

        // Record the total processing start time
        long totalStartTime = System.nanoTime();
        int processedFiles = 0;

        for (File svgFile : svgFiles) {
            String outputFileName = svgFile.getName().replace(".svg", ".png");
            File outputFile = new File(pngOutputFolder, outputFileName);

            try (FileInputStream svgInputStream = new FileInputStream(svgFile);
                 FileOutputStream pngOutputStream = new FileOutputStream(outputFile)) {

                // Create PNGTranscoder instance
                PNGTranscoder transcoder = new PNGTranscoder();

                // Set EXECUTE_ONLOAD to enable or disable onload script execution
               transcoder.addTranscodingHint(PNGTranscoder.KEY_EXECUTE_ONLOAD, Boolean.TRUE); // Enable onload execution
             //transcoder.addTranscodingHint(PNGTranscoder.KEY_EXECUTE_ONLOAD, Boolean.FALSE);

                // Set up the input and output for the transcoder
                TranscoderInput input = new TranscoderInput(svgInputStream);
                TranscoderOutput output = new TranscoderOutput(pngOutputStream);

                // Execute the conversion
                transcoder.transcode(input, output);
                processedFiles++;
                System.out.println("Processed file " + processedFiles + " / " + svgFiles.length);

            } catch (Exception e) {
                System.out.println("Error processing " + svgFile.getName() + ": " + e.getMessage());
            }
        }

        // Record the total processing end time
        long totalEndTime = System.nanoTime();

        // Calculate and print the total elapsed time in milliseconds
        long totalDuration = (totalEndTime - totalStartTime) / 1_000_000; // Convert nanoseconds to milliseconds
        System.out.println("All files have been processed! Total time: " + totalDuration + " milliseconds.");
    }
}