import org.apache.batik.transcoder.TranscoderInput;
import org.apache.batik.transcoder.TranscoderOutput;
import org.apache.batik.transcoder.image.PNGTranscoder;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class SnapshotTest {
    public static void main(String[] args) {
        String svgFolderPath = "/Users/wang/Documents/project/configuration_code_understanding/code4/data/snap_time/";
        String pngOutputFolderPath = "/Users/wang/Documents/project/configuration_code_understanding/code4/data/batik/output/";

        File svgFolder = new File(svgFolderPath);
        File[] svgFiles = svgFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".svg"));

        if (svgFiles == null || svgFiles.length == 0) {
            System.out.println("No SVG files found!");
            return;
        }

        File pngOutputFolder = new File(pngOutputFolderPath);
        if (!pngOutputFolder.exists()) {
            pngOutputFolder.mkdirs();
        }

        long totalStartTime = System.nanoTime();
        int processedFiles = 0;

        // Set different SNAPSHOT_TIME values for testing performance
        //float snapshotTime = 0f; // Beginning of the animation
        // float snapshotTime = 2f; // 2 seconds into the animation
        // float snapshotTime = 5f; // Midway through the animation
        // float snapshotTime = 7f; // 7 seconds into the animation
        // float snapshotTime = 10f; // End of a 10-second animation
        float snapshotTime = 20f;

        for (File svgFile : svgFiles) {
            String outputFileName = svgFile.getName().replace(".svg", "_snapshot_" + (int) snapshotTime + ".png");
            File outputFile = new File(pngOutputFolder, outputFileName);

            try (FileInputStream svgInputStream = new FileInputStream(svgFile);
                 FileOutputStream pngOutputStream = new FileOutputStream(outputFile)) {

                PNGTranscoder transcoder = new PNGTranscoder();

                // Set high resolution to increase rendering workload
                transcoder.addTranscodingHint(PNGTranscoder.KEY_WIDTH, 2000f);
                transcoder.addTranscodingHint(PNGTranscoder.KEY_HEIGHT, 2000f);

                // Set SNAPSHOT_TIME for the animation
                transcoder.addTranscodingHint(PNGTranscoder.KEY_SNAPSHOT_TIME, snapshotTime);

                TranscoderInput input = new TranscoderInput(svgInputStream);
                TranscoderOutput output = new TranscoderOutput(pngOutputStream);

                transcoder.transcode(input, output);
                processedFiles++;
                System.out.println("Processed file " + processedFiles + " / " + svgFiles.length);

            } catch (Exception e) {
                System.out.println("Error processing " + svgFile.getName() + ": " + e.getMessage());
            }
        }

        long totalEndTime = System.nanoTime();
        long totalDuration = (totalEndTime - totalStartTime) / 1_000_000;
        System.out.println("All files processed! Total time: " + totalDuration + " milliseconds.");
    }
}