import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;

/**
 * Goan Konkani Dialect Classifier - Java Command Line Interface (CLI)
 * Allows batch analysis and evaluation directly from Java.
 */
public class KonkaniDialectCLI {

    private static final String BACKEND_URL = "http://localhost:8000";

    public static void main(String[] args) {
        if (args.length == 0) {
            System.out.println("==========================================================================");
            System.out.println("   GOAN KONKANI DIALECT IDENTIFICATION USING SPEECH AUDIO - CLI");
            System.out.println("==========================================================================");
            System.out.println("Usage: java KonkaniDialectCLI <audio_file.wav> [model_choice]");
            System.out.println("  Models: ensemble (Dual v5+v6, default), tri (Tri-Engine v5+v6+v7), v7, v6, v5");
            System.out.println("\nExample:");
            System.out.println("  java KonkaniDialectCLI TEST_AUDIO_SAMPLES/01_Bardez_NorthGoa.wav ensemble");
            System.exit(0);
        }

        String audioPathStr = args[0];
        String modelChoice = args.length > 1 ? args[1] : "ensemble";

        Path audioPath = Path.of(audioPathStr);
        if (!Files.exists(audioPath)) {
            System.err.println("[ERROR] Audio file not found at " + audioPath.toAbsolutePath());
            System.exit(1);
        }

        System.out.println("Connecting to Konkani Dialect Classifier AI Model Server...");
        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();

        try {
            // Check health
            HttpRequest healthReq = HttpRequest.newBuilder()
                    .uri(URI.create(BACKEND_URL + "/health"))
                    .GET()
                    .build();
            HttpResponse<String> healthResp = client.send(healthReq, HttpResponse.BodyHandlers.ofString());
            if (healthResp.statusCode() != 200) {
                System.err.println("[ERROR] Backend server returned status " + healthResp.statusCode());
                System.exit(1);
            }

            System.out.println("[INFO] Backend connected! Uploading audio and executing MMS-300M inference...");
            long t0 = System.currentTimeMillis();

            // Build multipart request
            String boundary = "---" + System.currentTimeMillis();
            ByteArrayOutputStream baos = new ByteArrayOutputStream();

            // Determine MIME type
            String nameLower = audioPath.getFileName().toString().toLowerCase();
            String mime = "audio/wav";
            if (nameLower.endsWith(".mp3")) mime = "audio/mpeg";
            else if (nameLower.endsWith(".m4a")) mime = "audio/mp4";
            else if (nameLower.endsWith(".flac")) mime = "audio/flac";
            else if (nameLower.endsWith(".ogg")) mime = "audio/ogg";
            else if (nameLower.endsWith(".aac")) mime = "audio/aac";

            baos.write(("--" + boundary + "\r\n").getBytes());
            baos.write(("Content-Disposition: form-data; name=\"file\"; filename=\"" + audioPath.getFileName() + "\"\r\n").getBytes());
            baos.write(("Content-Type: " + mime + "\r\n\r\n").getBytes());
            baos.write(Files.readAllBytes(audioPath));
            baos.write("\r\n".getBytes());

            baos.write(("--" + boundary + "\r\n").getBytes());
            baos.write("Content-Disposition: form-data; name=\"model_choice\"\r\n\r\n".getBytes());
            baos.write((modelChoice + "\r\n").getBytes());

            baos.write(("--" + boundary + "--\r\n").getBytes());

            HttpRequest classifyReq = HttpRequest.newBuilder()
                    .uri(URI.create(BACKEND_URL + "/api/classify"))
                    .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                    .POST(HttpRequest.BodyPublishers.ofByteArray(baos.toByteArray()))
                    .timeout(Duration.ofSeconds(60))
                    .build();

            HttpResponse<String> resp = client.send(classifyReq, HttpResponse.BodyHandlers.ofString());
            long totalMs = System.currentTimeMillis() - t0;

            if (resp.statusCode() == 200) {
                renderCliCard(resp.body(), totalMs);
            } else {
                System.err.println("[ERROR] Classification failed: " + resp.body());
            }

        } catch (Exception e) {
            System.err.println("[ERROR] Connection Error: " + e.getMessage());
            System.err.println("Note: Make sure the Python backend is running: `python backend_server.py`");
        }
    }

    private static void renderCliCard(String json, long totalRoundtripMs) {
        String dialect = extractJsonString(json, "predicted_dialect");
        double confidence = extractJsonDouble(json, "confidence_percent");
        String region = extractJsonString(json, "region_name");
        String context = extractJsonString(json, "cultural_context");
        String badge = extractJsonString(json, "consensus_badge");
        double serverLatency = extractJsonDouble(json, "latency_ms");
        double duration = extractJsonDouble(json, "duration_seconds");
        String device = extractJsonString(json, "device");
        String engine = extractJsonString(json, "engine");

        System.out.println("\n==========================================================================================");
        System.out.println("                 GOAN KONKANI DIALECT CLASSIFICATION RESULT (JAVA CLIENT)                ");
        System.out.println("==========================================================================================");
        System.out.printf("  [RESULT] IDENTIFIED DIALECT:  %s\n", dialect.toUpperCase().replace("_", " "));
        System.out.printf("  [SCORE]  CONFIDENCE SCORE:    %.2f%%\n", confidence);
        System.out.printf("  [REGION] GEOGRAPHIC REGION:   %s\n", region);
        System.out.printf("  [STATUS] CONSENSUS STATUS:    %s\n", badge);
        System.out.printf("  [ENGINE] CLASSIFIER ENGINE:   %s\n", engine);
        System.out.println("------------------------------------------------------------------------------------------");
        System.out.println("  SOCIOLINGUISTIC CONTEXT:");
        System.out.println("     " + context);
        System.out.println("------------------------------------------------------------------------------------------");
        System.out.println("  DIALECT PROBABILITY DISTRIBUTION:");

        int pIdx = json.indexOf("\"probabilities\":{");
        if (pIdx != -1) {
            int endP = json.indexOf("}", pIdx);
            String pSub = json.substring(pIdx + 17, endP);
            String[] pairs = pSub.split(",");
            for (String pair : pairs) {
                String[] kv = pair.split(":");
                if (kv.length == 2) {
                    String k = kv[0].replace("\"", "").trim();
                    try {
                        double v = Double.parseDouble(kv[1].trim());
                        int barLength = (int) (v * 40);
                        String bar = "█".repeat(Math.max(1, barLength));
                        System.out.printf("     %-18s | %-40s | %6.2f%%\n", k.replace("_", " "), bar, v * 100);
                    } catch (Exception ignored) {
                    }
                }
            }
        }

        System.out.println("------------------------------------------------------------------------------------------");
        System.out.printf("  SERVER LATENCY:       %.2f ms (%s) | Speech Duration: %.2fs\n", serverLatency, device.toUpperCase(), duration);
        System.out.printf("  ROUND-TRIP LATENCY:   %d ms (Java HTTP Client -> Python FastAPI)\n", totalRoundtripMs);
        System.out.println("==========================================================================================\n");
    }

    private static String extractJsonString(String json, String key) {
        String target = "\"" + key + "\":\"";
        int idx = json.indexOf(target);
        if (idx == -1) return "";
        idx += target.length();
        int end = json.indexOf("\"", idx);
        if (end == -1) return "";
        return json.substring(idx, end).replace("\\\"", "\"");
    }

    private static double extractJsonDouble(String json, String key) {
        String target = "\"" + key + "\":";
        int idx = json.indexOf(target);
        if (idx == -1) return 0.0;
        idx += target.length();
        int end = idx;
        while (end < json.length() && (Character.isDigit(json.charAt(end)) || json.charAt(end) == '.' || json.charAt(end) == '-')) {
            end++;
        }
        try {
            return Double.parseDouble(json.substring(idx, end).trim());
        } catch (Exception e) {
            return 0.0;
        }
    }
}
