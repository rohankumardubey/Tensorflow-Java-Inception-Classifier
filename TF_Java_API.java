import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import org.tensorflow.*;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.io.File;
import java.io.*;
import java.nio.*;
import java.util.*;
import java.nio.file.Files;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.charset.Charset;

public class TF_Java_API{

    private static String imagepath = "images/test.jpg";
    private static String modelpath = "inception_dec_2015/";
    private static byte[] graphDef;
    private static byte[] imageBytes;
    private static List<String> labels;
    
    private static byte[] readAllBytesOrExit(Path path) {
        try {
            return Files.readAllBytes(path);
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(1);
        }
        return null;
    }
    
     private static List<String> readAllLinesOrExit(Path path) {
        try {
            return Files.readAllLines(path, Charset.forName("UTF-8"));
        } catch (IOException e) {
            System.err.println("Failed to read [" + path + "]: " + e.getMessage());
            System.exit(0);
        }
        return null;
    }
    
    private static int maxIndex(float[] probabilities) {
        int best = 0;
        for (int i = 1; i < probabilities.length; ++i) {
            if (probabilities[i] > probabilities[best]) {
                best = i;
            }
        }
        return best;
    }
    
    private static float[] executeInceptionGraph(byte[] graphDef, Tensor image) {
        try (Graph g = new Graph()) {
            g.importGraphDef(graphDef);
            try (Session s = new Session(g);
                    Tensor result = s.runner().feed("DecodeJpeg/contents", image).fetch("softmax").run().get(0)) {
                final long[] rshape = result.shape();
                if (result.numDimensions() != 2 || rshape[0] != 1) {
                    throw new RuntimeException(
                            String.format(
                                    "Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape %s",
                                    Arrays.toString(rshape)));
                }
                int nlabels = (int) rshape[1];
                float[][] temp = new float[1][nlabels];
                float[] labelProbabilities = ((float[][])result.copyTo(temp))[0];
                return labelProbabilities;
            }
        }
    }
    
    public static void main(String[] args) throws Exception {
        System.out.println("TensorFlow Version" + TensorFlow.version());
        
        graphDef = readAllBytesOrExit(Paths.get(modelpath, "tensorflow_inception_graph.pb"));
        imageBytes = readAllBytesOrExit(Paths.get(imagepath));
        labels = readAllLinesOrExit(Paths.get(modelpath, "imagenet_comp_graph_label_strings.txt"));
        
//         try(Tensor image = Tensor.create(imageBytes)){
//          executeInceptionGraph(graphDef, image);
//         }
        try (Tensor image = Tensor.create(imageBytes)) {
            float[] labelProbabilities = executeInceptionGraph(graphDef, image);
            int bestLabelIdx = maxIndex(labelProbabilities);
            System.out.println(
                    String.format(
                            "#### BEST MATCH: %s (%.2f%% Confidence) ####",
                            labels.get(bestLabelIdx), labelProbabilities[bestLabelIdx] * 100f));
            }
    }
}
