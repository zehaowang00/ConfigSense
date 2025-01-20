import java.io.File;
import java.io.FileWriter;
import java.io.IOException;

public class SvgGenerator {
    public static void main(String[] args) {
        String outputDirPath = "/Users/wang/Documents/project/configuration_code_understanding/code4/data/snap_time/";

        File outputDir = new File(outputDirPath);
        if (!outputDir.exists()) {
            outputDir.mkdirs();
        }

        for (int i = 1; i <= 50; i++) {
            String fileName = "complex_test_onload_" + i + ".svg";
            File svgFile = new File(outputDir, fileName);

            try (FileWriter writer = new FileWriter(svgFile)) {
                writer.write(generateComplexSvgContent(i));
                System.out.println("Created file: " + svgFile.getName());
            } catch (IOException e) {
                System.out.println("Error creating file: " + svgFile.getName());
                e.printStackTrace();
            }
        }

        System.out.println("50 complex SVG files generated successfully!");
    }

    private static String generateComplexSvgContent(int seed) {
        StringBuilder svgContent = new StringBuilder();
        svgContent.append("<svg width=\"500\" height=\"500\" xmlns=\"http://www.w3.org/2000/svg\">\n");
        svgContent.append("  <rect width=\"100%\" height=\"100%\" fill=\"white\" />\n");

        svgContent.append("  <script type=\"text/ecmascript\">\n");
        svgContent.append("    <![CDATA[\n");
        svgContent.append("      window.onload = function() {\n");
        svgContent.append("          function createCircles(time) {\n");
        svgContent.append("              for (let i = 0; i < 50; i++) {\n");
        svgContent.append("                  let circle = document.createElementNS(\"http://www.w3.org/2000/svg\", \"circle\");\n");
        svgContent.append("                  circle.setAttribute(\"cx\", Math.random() * 500);\n");
        svgContent.append("                  circle.setAttribute(\"cy\", Math.random() * 500);\n");
        svgContent.append("                  circle.setAttribute(\"r\", Math.random() * 30);\n");
        svgContent.append("                  circle.setAttribute(\"fill\", `rgb(${Math.random()*255}, ${Math.random()*255}, ${Math.random()*255})`);\n");

        // Add animation for radius
        svgContent.append("                  let animateR = document.createElementNS(\"http://www.w3.org/2000/svg\", \"animate\");\n");
        svgContent.append("                  animateR.setAttribute(\"attributeName\", \"r\");\n");
        svgContent.append("                  animateR.setAttribute(\"from\", 5);\n");
        svgContent.append("                  animateR.setAttribute(\"to\", 50);\n");
        svgContent.append("                  animateR.setAttribute(\"dur\", \"5s\");\n");
        svgContent.append("                  animateR.setAttribute(\"repeatCount\", \"indefinite\");\n");
        svgContent.append("                  circle.appendChild(animateR);\n");

        // Add animation for color
        svgContent.append("                  let animateColor = document.createElementNS(\"http://www.w3.org/2000/svg\", \"animate\");\n");
        svgContent.append("                  animateColor.setAttribute(\"attributeName\", \"fill\");\n");
        svgContent.append("                  animateColor.setAttribute(\"values\", `rgb(${Math.random()*255},${Math.random()*255},${Math.random()*255});rgb(${Math.random()*255},${Math.random()*255},${Math.random()*255})`);\n");
        svgContent.append("                  animateColor.setAttribute(\"dur\", \"5s\");\n");
        svgContent.append("                  animateColor.setAttribute(\"repeatCount\", \"indefinite\");\n");
        svgContent.append("                  circle.appendChild(animateColor);\n");

        svgContent.append("                  document.documentElement.appendChild(circle);\n");
        svgContent.append("              }\n");
        svgContent.append("          }\n");

        // Add circles every 2 seconds
        svgContent.append("          let times = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20];\n");
        svgContent.append("          times.forEach((time, index) => {\n");
        svgContent.append("              setTimeout(() => createCircles(time), time * 1000);\n");
        svgContent.append("          });\n");
        svgContent.append("      }\n");
        svgContent.append("    ]]>\n");
        svgContent.append("  </script>\n");

        svgContent.append("</svg>");
        return svgContent.toString();
    }
}