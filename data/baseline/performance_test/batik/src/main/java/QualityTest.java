import org.apache.batik.transcoder.TranscoderInput;
import org.apache.batik.transcoder.TranscoderOutput;
import org.apache.batik.transcoder.image.JPEGTranscoder;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;

public class QualityTest {
    public static void main(String[] args) {
        // 指定 SVG 文件夹路径
        String svgFolderPath = "/Users/wang/Documents/project/configuration_code_understanding/code4/data/icons/";
        // 指定 JPEG 输出文件夹路径
        String jpgOutputFolderPath = "/Users/wang/Documents/project/configuration_code_understanding/code4/data/batik/output/";

        File svgFolder = new File(svgFolderPath);
        File[] svgFiles = svgFolder.listFiles((dir, name) -> name.toLowerCase().endsWith(".svg"));

        if (svgFiles == null || svgFiles.length == 0) {
            System.out.println("没有找到 SVG 文件！");
            return;
        }

        // 创建 JPEG 输出文件夹
        File jpgOutputFolder = new File(jpgOutputFolderPath);
        if (!jpgOutputFolder.exists()) {
            jpgOutputFolder.mkdirs();
        }

        // 记录总处理开始时间
        long totalStartTime = System.nanoTime();
        int fileCount = 0;

        for (File svgFile : svgFiles) {
            String outputFileName = svgFile.getName().replace(".svg", ".jpg");
            File outputFile = new File(jpgOutputFolder, outputFileName);

            try (FileInputStream svgInputStream = new FileInputStream(svgFile);
                 FileOutputStream jpgOutputStream = new FileOutputStream(outputFile)) {

                // 创建 JPEG 转换器
                JPEGTranscoder transcoder = new JPEGTranscoder();

                // 配置 TranscodingHints：低质量或高质量
                // transcoder.addTranscodingHint(JPEGTranscoder.KEY_QUALITY, 0.1f); low
                // high quality
                 transcoder.addTranscodingHint(JPEGTranscoder.KEY_QUALITY, 1.0f); // high（1.0）

                // 设置输入输出
                TranscoderInput input = new TranscoderInput(svgInputStream);
                TranscoderOutput output = new TranscoderOutput(jpgOutputStream);

                // 执行转换
                transcoder.transcode(input, output);

                fileCount++;
                System.out.println("processed files: " + fileCount + " / " + svgFiles.length);

            } catch (Exception e) {
                System.out.println("process " + svgFile.getName() + " has issues：" + e.getMessage());
            }
        }

        // 记录总处理结束时间
        long totalEndTime = System.nanoTime();

        // 计算并打印总耗时
        long totalDuration = (totalEndTime - totalStartTime) / 1_000_000; // 转换为毫秒
        System.out.println("All files are transformed！The total coast time: " + totalDuration + " milliseconds");
    }
}