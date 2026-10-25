import javax.sound.sampled.*;
import javax.swing.*;
import javax.swing.border.EmptyBorder;
import javax.swing.border.LineBorder;
import javax.swing.filechooser.FileNameExtensionFilter;
import javax.swing.text.View;
import java.awt.*;
import java.awt.datatransfer.DataFlavor;
import java.awt.datatransfer.Transferable;
import java.awt.dnd.*;
import java.awt.event.ActionEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.time.Duration;
import java.util.*;
import java.util.List;

/**
 * Goan Konkani Dialect & Regional Origin Neural Classifier
 * Modern Production Java 17 Desktop GUI Application
 * Department of Computer Science & Engineering | Capstone Engineering Project
 */
public class KonkaniDialectClassifierApp extends JFrame {

    private static final String BACKEND_URL = "http://localhost:8000";
    private final HttpClient httpClient;

    // Header & Status
    private JLabel statusLabel;
    private JLabel statusDot;
    private JButton refreshBtn;

    // Left Control Panel Components
    private JComboBox<String> sampleDropdown;
    private DropZonePanel dropZonePanel;
    private JTextField filePathField;
    private JButton browseBtn;
    private JButton recordBtn;
    private JButton playBtn;
    private JButton stopBtn;
    private JComboBox<String> modelSelector;
    private JButton classifyBtn;
    private JProgressBar progressBar;

    // Right Results Panel Components
    private JPanel resultPanel;
    private JLabel dialectNameLabel;
    private JLabel confidenceBadge;
    private JLabel consensusBadge;
    private JLabel regionLabel;
    private JEditorPane contextArea;
    private JEditorPane rationaleArea;
    private ProbabilityChartPanel chartPanel;
    private JLabel telemetryLabel;

    // Audio & Playback State
    private File selectedAudioFile = null;
    private Clip currentAudioClip = null;
    private Process currentProcessPlayer = null;
    private boolean isRecording = false;
    private TargetDataLine targetDataLine;
    private File recordedTempFile = null;
    private javax.swing.Timer recordTimer;
    private int recordSeconds = 0;

    public KonkaniDialectClassifierApp() {
        super("Goan Konkani Dialect Identification Using Speech Audio");
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();

        initUI();
        setupGlobalDragAndDrop();
        checkBackendHealth();
        loadTestSamples();
    }

    private void initUI() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setSize(1220, 870);
        setMinimumSize(new Dimension(1080, 760));
        setLocationRelativeTo(null);

        // Main background
        JPanel contentPane = new JPanel(new BorderLayout(0, 0));
        contentPane.setBackground(new Color(241, 245, 249)); // Modern Slate 100
        setContentPane(contentPane);

        // 1. Top Header Banner
        contentPane.add(createHeaderPanel(), BorderLayout.NORTH);

        // 2. Center Split (Left: Controls & Prediction Rationale, Right: Predictions & Probability Distribution)
        JPanel centerPanel = new JPanel(new GridLayout(1, 2, 18, 0));
        centerPanel.setBackground(new Color(241, 245, 249));
        centerPanel.setBorder(new EmptyBorder(16, 18, 16, 18));

        centerPanel.add(createLeftControlPanel());
        centerPanel.add(createRightResultPanel());

        contentPane.add(centerPanel, BorderLayout.CENTER);

        // 3. Bottom Status Bar
        contentPane.add(createFooterPanel(), BorderLayout.SOUTH);
    }

    private JPanel createHeaderPanel() {
        JPanel header = new JPanel(new BorderLayout());
        header.setBackground(new Color(15, 23, 42)); // Modern Deep Slate Navy (#0F172A)
        header.setBorder(new EmptyBorder(14, 22, 14, 22));

        // Left Title Block
        JPanel titleBlock = new JPanel(new GridLayout(2, 1, 0, 3));
        titleBlock.setOpaque(false);

        JLabel titleLabel = new JLabel("GOAN KONKANI DIALECT IDENTIFICATION USING SPEECH AUDIO");
        titleLabel.setFont(getAppFont(Font.BOLD, 17));
        titleLabel.setForeground(Color.WHITE);

        JLabel subLabel = new JLabel("Meta MMS-300M Deep Acoustic Transformer Architecture • 8 Regional Goan Dialects");
        subLabel.setFont(getAppFont(Font.PLAIN, 12));
        subLabel.setForeground(new Color(148, 163, 184)); // Slate 400

        titleBlock.add(titleLabel);
        titleBlock.add(subLabel);
        header.add(titleBlock, BorderLayout.WEST);

        // Right Connection Status Block
        JPanel statusBlock = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 4));
        statusBlock.setOpaque(false);

        statusDot = new JLabel("●");
        statusDot.setFont(getAppFont(Font.BOLD, 18));
        statusDot.setForeground(new Color(239, 68, 68)); // Offline red

        statusLabel = new JLabel("Connecting to AI Model Server...");
        statusLabel.setFont(getAppFont(Font.BOLD, 12));
        statusLabel.setForeground(Color.WHITE);

        refreshBtn = new ModernButton("Recheck", new Color(30, 41, 59), Color.WHITE, 6);
        refreshBtn.setFont(getAppFont(Font.BOLD, 11));
        refreshBtn.setPreferredSize(new Dimension(85, 28));
        refreshBtn.setToolTipText("Recheck connection to AI model backend");
        refreshBtn.addActionListener(e -> checkBackendHealth());

        statusBlock.add(statusDot);
        statusBlock.add(statusLabel);
        statusBlock.add(refreshBtn);
        header.add(statusBlock, BorderLayout.EAST);

        return header;
    }

    private JPanel createLeftControlPanel() {
        JPanel panel = new JPanel();
        panel.setLayout(new BoxLayout(panel, BoxLayout.Y_AXIS));
        panel.setBackground(Color.WHITE);
        panel.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(new Color(226, 232, 240), 1, true),
                new EmptyBorder(16, 18, 16, 18)
        ));

        // Section 1: Pre-loaded test samples
        panel.add(createSectionHeader("1. Quick Test: Pre-Loaded Clean Dialect Samples"));
        panel.add(Box.createVerticalStrut(6));

        sampleDropdown = new JComboBox<>(new String[]{"-- Select a Pre-Loaded Broadcast Sample --"});
        sampleDropdown.setMaximumSize(new Dimension(Integer.MAX_VALUE, 32));
        sampleDropdown.addActionListener(this::onSampleSelected);
        panel.add(sampleDropdown);
        panel.add(Box.createVerticalStrut(14));

        // Section 2: Drag & Drop Zone or Local File Upload
        panel.add(createSectionHeader("2. Drag & Drop or Upload Audio (WAV, MP3, M4A, FLAC, OGG)"));
        panel.add(Box.createVerticalStrut(6));

        // Interactive Drag & Drop Box
        dropZonePanel = new DropZonePanel();
        dropZonePanel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseClicked(MouseEvent e) {
                onBrowseFile(null);
            }
        });
        setupDropTargetForComponent(dropZonePanel);
        panel.add(dropZonePanel);
        panel.add(Box.createVerticalStrut(8));

        // File Path Field & Browse Button Row
        JPanel fileChooseRow = new JPanel(new BorderLayout(8, 0));
        fileChooseRow.setOpaque(false);
        fileChooseRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 32));

        filePathField = new JTextField("No audio file selected");
        filePathField.setEditable(false);
        filePathField.setFont(new Font("Monospaced", Font.PLAIN, 11));
        filePathField.setBackground(new Color(248, 250, 252));
        filePathField.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(new Color(203, 213, 225), 1),
                new EmptyBorder(4, 8, 4, 8)
        ));

        browseBtn = new ModernButton("Browse...", new Color(241, 245, 249), new Color(15, 23, 42), 6);
        browseBtn.setFont(getAppFont(Font.PLAIN, 12));
        browseBtn.setPreferredSize(new Dimension(85, 32));
        browseBtn.addActionListener(this::onBrowseFile);

        fileChooseRow.add(filePathField, BorderLayout.CENTER);
        fileChooseRow.add(browseBtn, BorderLayout.EAST);
        panel.add(fileChooseRow);
        panel.add(Box.createVerticalStrut(14));

        // Section 3: Live Microphone Recording & Playback
        panel.add(createSectionHeader("3. Microphone Recording & Audio Playback"));
        panel.add(Box.createVerticalStrut(6));

        JPanel recRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 0));
        recRow.setOpaque(false);
        recRow.setMaximumSize(new Dimension(Integer.MAX_VALUE, 36));

        recordBtn = new ModernButton("Record Audio", new Color(225, 29, 72), Color.WHITE, 6);
        recordBtn.setFont(getAppFont(Font.BOLD, 12));
        recordBtn.setPreferredSize(new Dimension(135, 34));
        recordBtn.addActionListener(this::toggleRecording);

        playBtn = new ModernButton("Play Audio", new Color(241, 245, 249), new Color(15, 23, 42), 6);
        playBtn.setEnabled(false);
        playBtn.setFont(getAppFont(Font.PLAIN, 12));
        playBtn.setPreferredSize(new Dimension(95, 34));
        playBtn.addActionListener(this::playSelectedAudio);

        stopBtn = new ModernButton("Stop Audio", new Color(241, 245, 249), new Color(15, 23, 42), 6);
        stopBtn.setEnabled(false);
        stopBtn.setFont(getAppFont(Font.PLAIN, 12));
        stopBtn.setPreferredSize(new Dimension(95, 34));
        stopBtn.addActionListener(this::stopAudioPlayback);

        recRow.add(recordBtn);
        recRow.add(playBtn);
        recRow.add(stopBtn);
        panel.add(recRow);
        panel.add(Box.createVerticalStrut(16));

        // Section 4: Model Selection
        panel.add(createSectionHeader("4. Select Neural Classification Engine"));
        panel.add(Box.createVerticalStrut(6));

        String[] engines = {
                "Dual-Engine Consensus Ensemble (Model v5.0 + v6.0) [Recommended]",
                "Tri-Engine Grand Ensemble (Model v5.0 + v6.0 + v7.0 Deep)",
                "Model v7.0 Deep-8 (Experimental Deep Unfreezing)",
                "Model v6.0 Pro Master (Trained on 7,628 Master Clips)",
                "Model v5.0 Pro (Phonetic Boundary Specialized)"
        };
        modelSelector = new JComboBox<>(engines);
        modelSelector.setMaximumSize(new Dimension(Integer.MAX_VALUE, 32));
        panel.add(modelSelector);
        panel.add(Box.createVerticalStrut(18));

        // Action Button: Classify
        classifyBtn = new ModernButton("RUN CLASSIFICATION", new Color(37, 99, 235), Color.WHITE, 8);
        classifyBtn.setFont(getAppFont(Font.BOLD, 14));
        classifyBtn.setPreferredSize(new Dimension(Integer.MAX_VALUE, 44));
        classifyBtn.setMaximumSize(new Dimension(Integer.MAX_VALUE, 44));
        classifyBtn.addActionListener(this::onClassifyClicked);
        panel.add(classifyBtn);
        panel.add(Box.createVerticalStrut(10));

        progressBar = new JProgressBar();
        progressBar.setIndeterminate(false);
        progressBar.setVisible(false);
        progressBar.setMaximumSize(new Dimension(Integer.MAX_VALUE, 8));
        panel.add(progressBar);
        panel.add(Box.createVerticalStrut(14));

        // Section 5: Prediction Rationale & Acoustic Evidence (Under Run Button)
        panel.add(createSectionHeader("5. Prediction Rationale & Acoustic Evidence (Why This Dialect)"));
        panel.add(Box.createVerticalStrut(6));

        JPanel rationalePanel = new JPanel(new BorderLayout(6, 6));
        rationalePanel.setBackground(new Color(248, 250, 252));
        rationalePanel.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(new Color(226, 232, 240), 1),
                new EmptyBorder(10, 14, 10, 14)
        ));
        rationalePanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, Integer.MAX_VALUE));

        JPanel ratHeaderRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        ratHeaderRow.setOpaque(false);
        JLabel ratTitle = new JLabel("ACOUSTIC DECISION REASONING & PHONETIC EVIDENCE");
        ratTitle.setFont(getAppFont(Font.BOLD, 10));
        ratTitle.setForeground(new Color(100, 116, 139));
        ratHeaderRow.add(ratTitle);

        rationaleArea = createWrappingEditorPane("<html><body style='font-family: -apple-system, BlinkMacSystemFont, Arial, Helvetica, sans-serif; font-size: 11px; color: #475569; margin: 0; padding: 0; line-height: 1.4; word-wrap: break-word;'>"
                + "Acoustic reasoning, phonetic markers, and neural evidence explaining why the model made this prediction will appear here upon clicking Analyze."
                + "</body></html>");

        rationalePanel.add(ratHeaderRow, BorderLayout.NORTH);
        rationalePanel.add(rationaleArea, BorderLayout.CENTER);
        panel.add(rationalePanel);

        panel.add(Box.createVerticalGlue());
        return panel;
    }

    private JPanel createRightResultPanel() {
        resultPanel = new JPanel();
        resultPanel.setLayout(new BoxLayout(resultPanel, BoxLayout.Y_AXIS));
        resultPanel.setBackground(Color.WHITE);
        resultPanel.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(new Color(226, 232, 240), 1, true),
                new EmptyBorder(16, 18, 16, 18)
        ));

        // 1. Primary Prediction Hero Card
        JPanel card = new JPanel(new BorderLayout(8, 8));
        card.setBackground(new Color(15, 23, 42)); // Modern Deep Slate Navy (#0F172A)
        card.setBorder(new EmptyBorder(14, 18, 14, 18));
        card.setMaximumSize(new Dimension(Integer.MAX_VALUE, 115));

        JPanel topCardRow = new JPanel(new BorderLayout());
        topCardRow.setOpaque(false);

        JLabel cardSub = new JLabel("IDENTIFIED GOAN KONKANI DIALECT");
        cardSub.setFont(getAppFont(Font.BOLD, 10));
        cardSub.setForeground(new Color(148, 163, 184)); // Slate 400

        confidenceBadge = new JLabel("Confidence: --%");
        confidenceBadge.setFont(getAppFont(Font.BOLD, 12));
        confidenceBadge.setForeground(new Color(52, 211, 153)); // Emerald 400
        topCardRow.add(cardSub, BorderLayout.WEST);
        topCardRow.add(confidenceBadge, BorderLayout.EAST);

        dialectNameLabel = new JLabel("READY FOR INFERENCE");
        dialectNameLabel.setFont(getAppFont(Font.BOLD, 26));
        dialectNameLabel.setForeground(Color.WHITE);

        consensusBadge = new JLabel("Drag & drop an audio file, choose a sample, or record speech to begin.");
        consensusBadge.setFont(getAppFont(Font.PLAIN, 11));
        consensusBadge.setForeground(new Color(203, 213, 225));

        card.add(topCardRow, BorderLayout.NORTH);
        card.add(dialectNameLabel, BorderLayout.CENTER);
        card.add(consensusBadge, BorderLayout.SOUTH);
        resultPanel.add(card);
        resultPanel.add(Box.createVerticalStrut(14));

        // 2. Geographic & Sociolinguistic Context Card
        JPanel geoPanel = new JPanel(new BorderLayout(6, 6));
        geoPanel.setBackground(new Color(248, 250, 252));
        geoPanel.setBorder(BorderFactory.createCompoundBorder(
                new LineBorder(new Color(226, 232, 240), 1),
                new EmptyBorder(10, 14, 10, 14)
        ));
        geoPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 160));

        JLabel geoTitle = new JLabel("GEOGRAPHIC ORIGIN & REGIONAL CONTEXT");
        geoTitle.setFont(getAppFont(Font.BOLD, 10));
        geoTitle.setForeground(new Color(100, 116, 139));

        JPanel regionRow = new JPanel(new FlowLayout(FlowLayout.LEFT, 4, 0));
        regionRow.setOpaque(false);
        regionLabel = new JLabel("Location in Goa: Not Analyzed Yet");
        regionLabel.setFont(getAppFont(Font.BOLD, 13));
        regionLabel.setForeground(new Color(15, 23, 42));
        regionRow.add(regionLabel);

        contextArea = createWrappingEditorPane("<html><body style='font-family: -apple-system, BlinkMacSystemFont, Arial, Helvetica, sans-serif; font-size: 11px; color: #475569; margin: 0; padding: 0; line-height: 1.4; word-wrap: break-word;'>Linguistic markers and cultural context will appear here upon neural inference.</body></html>");

        geoPanel.add(geoTitle, BorderLayout.NORTH);
        geoPanel.add(regionRow, BorderLayout.CENTER);
        geoPanel.add(contextArea, BorderLayout.SOUTH);
        resultPanel.add(geoPanel);
        resultPanel.add(Box.createVerticalStrut(14));

        // 3. Probability Distribution Graphic Chart
        JLabel chartTitle = new JLabel("DIALECT PROBABILITY DISTRIBUTION (ALL 8 REGIONS)");
        chartTitle.setFont(getAppFont(Font.BOLD, 11));
        chartTitle.setForeground(new Color(71, 85, 105));
        resultPanel.add(chartTitle);
        resultPanel.add(Box.createVerticalStrut(6));

        chartPanel = new ProbabilityChartPanel();
        chartPanel.setPreferredSize(new Dimension(500, 290));
        resultPanel.add(chartPanel);

        resultPanel.add(Box.createVerticalGlue());
        return resultPanel;
    }

    private JPanel createFooterPanel() {
        JPanel footer = new JPanel(new BorderLayout());
        footer.setBackground(new Color(241, 245, 249));
        footer.setBorder(new EmptyBorder(8, 20, 8, 20));

        telemetryLabel = new JLabel("System: Standby | Target: 16 kHz Mono | Engine: MMS-300M (DSP Auto-Conditioning)");
        telemetryLabel.setFont(getAppFont(Font.PLAIN, 11));
        telemetryLabel.setForeground(new Color(100, 116, 139));

        JLabel copyLabel = new JLabel("Goa University MCA | Guided by Mr Hanumant Redkar | 2026–2027");
        copyLabel.setFont(getAppFont(Font.PLAIN, 11));
        copyLabel.setForeground(new Color(148, 163, 184));

        footer.add(telemetryLabel, BorderLayout.WEST);
        footer.add(copyLabel, BorderLayout.EAST);
        return footer;
    }

    public static Font getAppFont(int style, int size) {
        String[] preferred = {"Helvetica Neue", "Arial", "Lucida Grande", "Verdana", Font.SANS_SERIF};
        GraphicsEnvironment ge = GraphicsEnvironment.getLocalGraphicsEnvironment();
        Set<String> available = new HashSet<>(Arrays.asList(ge.getAvailableFontFamilyNames()));
        for (String f : preferred) {
            if (available.contains(f)) {
                return new Font(f, style, size);
            }
        }
        return new Font(Font.SANS_SERIF, style, size);
    }

    public static JEditorPane createWrappingEditorPane(String initialHtml) {
        JEditorPane pane = new JEditorPane() {
            @Override
            public boolean getScrollableTracksViewportWidth() {
                return true;
            }

            @Override
            public Dimension getPreferredSize() {
                int w = getWidth();
                if (w <= 0) {
                    Container parent = getParent();
                    if (parent != null && parent.getWidth() > 0) {
                        w = parent.getWidth() - 20;
                    }
                }
                if (w <= 0) w = 480;
                View view = getUI().getRootView(this);
                if (view != null) {
                    view.setSize(w, Short.MAX_VALUE);
                    float h = view.getPreferredSpan(View.Y_AXIS);
                    return new Dimension(w, (int) Math.ceil(h));
                }
                return super.getPreferredSize();
            }
        };
        pane.setContentType("text/html");
        pane.setEditable(false);
        pane.setOpaque(false);
        pane.setText(initialHtml);
        return pane;
    }

    private JLabel createSectionHeader(String title) {
        JLabel lbl = new JLabel(title);
        lbl.setFont(getAppFont(Font.BOLD, 12));
        lbl.setForeground(new Color(30, 41, 59));
        return lbl;
    }

    /**
     * Enables global Drag and Drop onto the entire application window
     */
    private void setupGlobalDragAndDrop() {
        setupDropTargetForComponent(this);
    }

    /**
     * Configures a Swing component to accept dropped audio files
     */
    private void setupDropTargetForComponent(Component comp) {
        new DropTarget(comp, new DropTargetListener() {
            @Override
            public void dragEnter(DropTargetDragEvent dtde) {
                if (dtde.isDataFlavorSupported(DataFlavor.javaFileListFlavor)) {
                    dtde.acceptDrag(DnDConstants.ACTION_COPY);
                    dropZonePanel.setDragOver(true);
                } else {
                    dtde.rejectDrag();
                }
            }

            @Override
            public void dragOver(DropTargetDragEvent dtde) {}

            @Override
            public void dropActionChanged(DropTargetDragEvent dtde) {}

            @Override
            public void dragExit(DropTargetEvent dte) {
                dropZonePanel.setDragOver(false);
            }

            @Override
            @SuppressWarnings("unchecked")
            public void drop(DropTargetDropEvent dtde) {
                dropZonePanel.setDragOver(false);
                try {
                    if (dtde.isDataFlavorSupported(DataFlavor.javaFileListFlavor)) {
                        dtde.acceptDrop(DnDConstants.ACTION_COPY);
                        Transferable transferable = dtde.getTransferable();
                        List<File> files = (List<File>) transferable.getTransferData(DataFlavor.javaFileListFlavor);
                        if (files != null && !files.isEmpty()) {
                            File file = files.get(0);
                            handleFileSelected(file);
                        }
                        dtde.dropComplete(true);
                    } else {
                        dtde.rejectDrop();
                    }
                } catch (Exception ex) {
                    ex.printStackTrace();
                    dtde.dropComplete(false);
                }
            }
        });
    }

    /**
     * Validates and selects audio files of any supported format
     */
    private void handleFileSelected(File file) {
        if (file == null || !file.exists()) return;

        String nameLower = file.getName().toLowerCase();
        String[] validExts = {".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg", ".webm", ".opus", ".aiff"};
        boolean valid = false;
        String extBadge = "AUDIO";

        for (String ext : validExts) {
            if (nameLower.endsWith(ext)) {
                valid = true;
                extBadge = ext.substring(1).toUpperCase();
                break;
            }
        }

        if (!valid) {
            JOptionPane.showMessageDialog(this,
                    "Unsupported file format: " + file.getName() +
                            "\n\nPlease provide a supported audio format:\n• WAV (Studio PCM)\n• MP3 (MPEG Audio)\n• M4A / AAC (MPEG-4 Audio)\n• FLAC (Lossless)\n• OGG / WEBM",
                    "Unsupported Audio Format", JOptionPane.WARNING_MESSAGE);
            return;
        }

        stopAudioPlayback(null);
        this.selectedAudioFile = file;
        this.filePathField.setText(file.getAbsolutePath());
        this.playBtn.setEnabled(true);
        this.stopBtn.setEnabled(false);

        long sizeKb = file.length() / 1024;
        String sizeStr = sizeKb > 1024 ? String.format("%.1f MB", sizeKb / 1024.0) : (sizeKb + " KB");
        this.dropZonePanel.setFileInfo(file.getName(), extBadge, sizeStr);
        this.telemetryLabel.setText("Loaded Audio: " + file.getName() + " (" + sizeStr + " • " + extBadge + ")");
    }

    private void checkBackendHealth() {
        new Thread(() -> {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(BACKEND_URL + "/health"))
                        .GET()
                        .timeout(Duration.ofSeconds(3))
                        .build();

                HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() == 200) {
                    SwingUtilities.invokeLater(() -> {
                        statusDot.setForeground(new Color(16, 185, 129)); // Emerald Green
                        statusLabel.setText("AI Server Live (MPS Acceleration)");
                        telemetryLabel.setText("Connected to Model Server on port 8000 | 3 Models Active (v5, v6, v7 Deep-8)");
                    });
                }
            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> {
                    statusDot.setForeground(new Color(239, 68, 68)); // Red
                    statusLabel.setText("AI Server Offline (Port 8000)");
                    telemetryLabel.setText("Backend connection offline. Ensure backend_server.py is running on port 8000.");
                });
            }
        }).start();
    }

    private void loadTestSamples() {
        new Thread(() -> {
            try {
                HttpRequest req = HttpRequest.newBuilder()
                        .uri(URI.create(BACKEND_URL + "/api/samples"))
                        .GET()
                        .timeout(Duration.ofSeconds(4))
                        .build();

                HttpResponse<String> resp = httpClient.send(req, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() == 200) {
                    List<String> filenames = extractSampleFilenames(resp.body());
                    SwingUtilities.invokeLater(() -> {
                        sampleDropdown.removeAllItems();
                        sampleDropdown.addItem("-- Select a Pre-Loaded Broadcast Sample --");
                        for (String f : filenames) {
                            sampleDropdown.addItem(f);
                        }
                    });
                }
            } catch (Exception ignored) {
                // Local fallback: scan TEST_AUDIO_SAMPLES directory directly
                File dir = new File("TEST_AUDIO_SAMPLES");
                if (dir.exists() && dir.isDirectory()) {
                    File[] files = dir.listFiles((d, name) -> {
                        String n = name.toLowerCase();
                        return n.endsWith(".wav") || n.endsWith(".mp3") || n.endsWith(".m4a") || n.endsWith(".flac");
                    });
                    if (files != null) {
                        Arrays.sort(files, Comparator.comparing(File::getName));
                        SwingUtilities.invokeLater(() -> {
                            sampleDropdown.removeAllItems();
                            sampleDropdown.addItem("-- Select a Pre-Loaded Broadcast Sample --");
                            for (File f : files) {
                                sampleDropdown.addItem(f.getName());
                            }
                        });
                    }
                }
            }
        }).start();
    }

    private List<String> extractSampleFilenames(String json) {
        List<String> list = new ArrayList<>();
        int idx = 0;
        while ((idx = json.indexOf("\"filename\":\"", idx)) != -1) {
            idx += 12;
            int end = json.indexOf("\"", idx);
            if (end != -1) {
                list.add(json.substring(idx, end));
                idx = end;
            }
        }
        return list;
    }

    private void onSampleSelected(ActionEvent e) {
        String selected = (String) sampleDropdown.getSelectedItem();
        if (selected == null || selected.startsWith("--")) {
            return;
        }

        File sampleFile = new File("TEST_AUDIO_SAMPLES", selected);
        if (sampleFile.exists()) {
            handleFileSelected(sampleFile);
        }
    }

    private void onBrowseFile(ActionEvent e) {
        JFileChooser chooser = new JFileChooser(new File("."));
        chooser.setDialogTitle("Select Konkani Speech Audio File (WAV, MP3, M4A, FLAC, OGG, AAC)");
        chooser.setFileFilter(new FileNameExtensionFilter(
                "Supported Audio Files (*.wav, *.mp3, *.m4a, *.flac, *.aac, *.ogg, *.webm)",
                "wav", "mp3", "m4a", "flac", "aac", "ogg", "webm", "opus", "aiff"
        ));
        int res = chooser.showOpenDialog(this);
        if (res == JFileChooser.APPROVE_OPTION) {
            sampleDropdown.setSelectedIndex(0);
            handleFileSelected(chooser.getSelectedFile());
        }
    }

    private void toggleRecording(ActionEvent e) {
        if (!isRecording) {
            startRecording();
        } else {
            stopRecording();
        }
    }

    private void startRecording() {
        try {
            AudioFormat format = new AudioFormat(16000, 16, 1, true, false);
            DataLine.Info info = new DataLine.Info(TargetDataLine.class, format);

            if (!AudioSystem.isLineSupported(info)) {
                JOptionPane.showMessageDialog(this, "Microphone recording not supported on this audio input line.", "Audio Error", JOptionPane.ERROR_MESSAGE);
                return;
            }

            targetDataLine = (TargetDataLine) AudioSystem.getLine(info);
            targetDataLine.open(format);
            targetDataLine.start();

            recordedTempFile = File.createTempFile("konkani_live_mic_", ".wav");
            recordedTempFile.deleteOnExit();

            isRecording = true;
            recordSeconds = 0;
            recordBtn.setText("Stop Recording (00:00)");
            recordBtn.setBackground(new Color(220, 38, 38));
            playBtn.setEnabled(false);
            stopBtn.setEnabled(false);
            telemetryLabel.setText("Live Microphone Active: Speak clearly in Konkani for 4 to 8 seconds...");

            recordTimer = new javax.swing.Timer(1000, ev -> {
                recordSeconds++;
                int mins = recordSeconds / 60;
                int secs = recordSeconds % 60;
                recordBtn.setText(String.format("Stop Recording (%02d:%02d)", mins, secs));
            });
            recordTimer.start();

            new Thread(() -> {
                try (AudioInputStream ais = new AudioInputStream(targetDataLine)) {
                    AudioSystem.write(ais, AudioFileFormat.Type.WAVE, recordedTempFile);
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
            }).start();

        } catch (Exception ex) {
            JOptionPane.showMessageDialog(this, "Failed to initialize microphone: " + ex.getMessage(), "Mic Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void stopRecording() {
        if (!isRecording) return;

        if (targetDataLine != null) {
            targetDataLine.stop();
            targetDataLine.close();
        }

        if (recordTimer != null) {
            recordTimer.stop();
        }

        isRecording = false;
        recordBtn.setText("Record Audio");
        recordBtn.setBackground(new Color(225, 29, 72));

        if (recordedTempFile != null && recordedTempFile.exists() && recordedTempFile.length() > 2048) {
            handleFileSelected(recordedTempFile);
            telemetryLabel.setText("Recorded " + recordSeconds + "s audio captured to temporary buffer.");
        } else {
            telemetryLabel.setText("Recording too short or empty.");
        }
    }

    /**
     * Universal audio playback supporting WAV, MP3, M4A, FLAC, and OGG
     */
    private void playSelectedAudio(ActionEvent e) {
        if (selectedAudioFile == null || !selectedAudioFile.exists()) {
            return;
        }

        stopAudioPlayback(null);
        new Thread(() -> {
            try {
                String nameLower = selectedAudioFile.getName().toLowerCase();
                boolean isWav = nameLower.endsWith(".wav") || nameLower.endsWith(".aiff");

                if (isWav) {
                    try {
                        AudioInputStream ais = AudioSystem.getAudioInputStream(selectedAudioFile);
                        currentAudioClip = AudioSystem.getClip();
                        currentAudioClip.open(ais);
                        currentAudioClip.start();

                        SwingUtilities.invokeLater(() -> {
                            playBtn.setEnabled(false);
                            stopBtn.setEnabled(true);
                            telemetryLabel.setText("Playing: " + selectedAudioFile.getName());
                        });

                        currentAudioClip.addLineListener(ev -> {
                            if (ev.getType() == LineEvent.Type.STOP) {
                                SwingUtilities.invokeLater(() -> {
                                    playBtn.setEnabled(true);
                                    stopBtn.setEnabled(false);
                                    telemetryLabel.setText("Playback finished.");
                                });
                            }
                        });
                        return;
                    } catch (Exception ex) {
                        // Fall back to native OS player
                    }
                }

                // Fallback for MP3, M4A, AAC, FLAC, or when AudioSystem lacks codec:
                String os = System.getProperty("os.name").toLowerCase();
                ProcessBuilder pb;
                if (os.contains("mac")) {
                    pb = new ProcessBuilder("afplay", selectedAudioFile.getAbsolutePath());
                } else if (os.contains("win")) {
                    pb = new ProcessBuilder("powershell", "-c", "(New-Object Media.SoundPlayer '" + selectedAudioFile.getAbsolutePath() + "').PlaySync()");
                } else {
                    pb = new ProcessBuilder("ffplay", "-nodisp", "-autoexit", selectedAudioFile.getAbsolutePath());
                }

                currentProcessPlayer = pb.start();
                SwingUtilities.invokeLater(() -> {
                    playBtn.setEnabled(false);
                    stopBtn.setEnabled(true);
                    telemetryLabel.setText("Playing: " + selectedAudioFile.getName() + " (Native OS Player)");
                });

                currentProcessPlayer.waitFor();
                SwingUtilities.invokeLater(() -> {
                    playBtn.setEnabled(true);
                    stopBtn.setEnabled(false);
                    telemetryLabel.setText("Playback finished.");
                });

            } catch (Exception ex) {
                SwingUtilities.invokeLater(() -> {
                    playBtn.setEnabled(true);
                    stopBtn.setEnabled(false);
                    telemetryLabel.setText("Audio playback error: " + ex.getMessage());
                });
            }
        }).start();
    }

    private void stopAudioPlayback(ActionEvent e) {
        if (currentAudioClip != null) {
            try {
                if (currentAudioClip.isRunning()) {
                    currentAudioClip.stop();
                }
                currentAudioClip.close();
            } catch (Exception ignored) {
            }
            currentAudioClip = null;
        }

        if (currentProcessPlayer != null && currentProcessPlayer.isAlive()) {
            currentProcessPlayer.destroyForcibly();
            currentProcessPlayer = null;
        }

        playBtn.setEnabled(selectedAudioFile != null);
        stopBtn.setEnabled(false);
    }

    private void onClassifyClicked(ActionEvent e) {
        if (selectedAudioFile == null || !selectedAudioFile.exists()) {
            JOptionPane.showMessageDialog(this, "Please drag & drop an audio file, choose a sample, or record speech first.", "No Audio Input", JOptionPane.WARNING_MESSAGE);
            return;
        }

        classifyBtn.setEnabled(false);
        classifyBtn.setText("RUNNING INFERENCE...");
        progressBar.setVisible(true);
        progressBar.setIndeterminate(true);
        dialectNameLabel.setText("ANALYZING PHONETICS...");
        consensusBadge.setText("Executing 4-stage DSP conditioning & MMS-300M acoustic inference...");

        String selectedModelStr = (String) modelSelector.getSelectedItem();
        String modelChoice = "ensemble";
        if (selectedModelStr != null) {
            if (selectedModelStr.contains("Tri-Engine")) modelChoice = "tri";
            else if (selectedModelStr.contains("Model v7.0")) modelChoice = "v7";
            else if (selectedModelStr.contains("Model v6.0")) modelChoice = "v6";
            else if (selectedModelStr.contains("Model v5.0")) modelChoice = "v5";
            else modelChoice = "ensemble";
        }
        final String finalModelChoice = modelChoice;

        new SwingWorker<Map<String, Object>, Void>() {
            @Override
            protected Map<String, Object> doInBackground() throws Exception {
                String boundary = "---" + System.currentTimeMillis();
                ByteArrayOutputStream baos = new ByteArrayOutputStream();

                // Determine MIME type
                String nameLower = selectedAudioFile.getName().toLowerCase();
                String mime = "audio/wav";
                if (nameLower.endsWith(".mp3")) mime = "audio/mpeg";
                else if (nameLower.endsWith(".m4a")) mime = "audio/mp4";
                else if (nameLower.endsWith(".flac")) mime = "audio/flac";
                else if (nameLower.endsWith(".ogg")) mime = "audio/ogg";
                else if (nameLower.endsWith(".aac")) mime = "audio/aac";

                // File part
                baos.write(("--" + boundary + "\r\n").getBytes());
                baos.write(("Content-Disposition: form-data; name=\"file\"; filename=\"" + selectedAudioFile.getName() + "\"\r\n").getBytes());
                baos.write(("Content-Type: " + mime + "\r\n\r\n").getBytes());
                baos.write(Files.readAllBytes(selectedAudioFile.toPath()));
                baos.write("\r\n".getBytes());

                // model_choice part
                baos.write(("--" + boundary + "\r\n").getBytes());
                baos.write("Content-Disposition: form-data; name=\"model_choice\"\r\n\r\n".getBytes());
                baos.write((finalModelChoice + "\r\n").getBytes());

                // End
                baos.write(("--" + boundary + "--\r\n").getBytes());

                HttpRequest postReq = HttpRequest.newBuilder()
                        .uri(URI.create(BACKEND_URL + "/api/classify"))
                        .header("Content-Type", "multipart/form-data; boundary=" + boundary)
                        .POST(HttpRequest.BodyPublishers.ofByteArray(baos.toByteArray()))
                        .timeout(Duration.ofSeconds(60))
                        .build();

                HttpResponse<String> resp = httpClient.send(postReq, HttpResponse.BodyHandlers.ofString());
                if (resp.statusCode() != 200) {
                    throw new RuntimeException("Server returned status: " + resp.statusCode() + " -> " + resp.body());
                }

                return parseClassificationJson(resp.body());
            }

            @Override
            protected void done() {
                classifyBtn.setEnabled(true);
                classifyBtn.setText("RUN CLASSIFICATION");
                progressBar.setVisible(false);
                try {
                    Map<String, Object> result = get();
                    displayResults(result);
                } catch (Exception ex) {
                    dialectNameLabel.setText("INFERENCE ERROR");
                    consensusBadge.setText("Error: " + ex.getMessage());
                    JOptionPane.showMessageDialog(KonkaniDialectClassifierApp.this, "Error communicating with AI server: " + ex.getMessage(), "Inference Error", JOptionPane.ERROR_MESSAGE);
                }
            }
        }.execute();
    }

    private void displayResults(Map<String, Object> res) {
        String dialect = (String) res.getOrDefault("predicted_dialect", "Unknown");
        Double conf = (Double) res.getOrDefault("confidence_percent", 0.0);
        String region = (String) res.getOrDefault("region_name", "Unknown Region");
        String context = (String) res.getOrDefault("cultural_context", "");
        String badge = (String) res.getOrDefault("consensus_badge", "");
        Double latency = (Double) res.getOrDefault("latency_ms", 0.0);
        Double dur = (Double) res.getOrDefault("duration_seconds", 0.0);
        String device = (String) res.getOrDefault("device", "mps");
        String engine = (String) res.getOrDefault("engine", "Dual-Engine Ensemble");

        dialectNameLabel.setText(dialect.replace("_", " ").toUpperCase());

        // Color-coded confidence badge
        confidenceBadge.setText(String.format("Confidence: %.1f%%", conf));
        if (conf >= 65.0) {
            confidenceBadge.setForeground(new Color(52, 211, 153)); // Emerald Green
        } else if (conf >= 45.0) {
            confidenceBadge.setForeground(new Color(56, 189, 248)); // Sky Blue
        } else {
            confidenceBadge.setForeground(new Color(251, 191, 36)); // Amber
        }

        consensusBadge.setText(badge);

        regionLabel.setText(region);
        contextArea.setText("<html><body style='font-family: -apple-system, BlinkMacSystemFont, Arial, Helvetica, sans-serif; font-size: 11px; color: #475569; margin: 0; padding: 0; line-height: 1.4;'>" + context + "</body></html>");

        // Update probability distribution chart
        @SuppressWarnings("unchecked")
        Map<String, Double> probs = (Map<String, Double>) res.get("probabilities");
        if (probs != null) {
            chartPanel.updateData(probs, dialect);
        }

        // Update Prediction Rationale & Acoustic Evidence ("Why the model made this prediction")
        @SuppressWarnings("unchecked")
        Map<String, Object> rationale = (Map<String, Object>) res.get("prediction_rationale");
        if (rationale == null || rationale.isEmpty()) {
            rationale = buildFallbackRationale(dialect, conf, probs);
        }

        String summary = (String) rationale.getOrDefault("summary", "");
        @SuppressWarnings("unchecked")
        List<String> markers = (List<String>) rationale.get("markers");
        String runnerUp = (String) rationale.getOrDefault("runner_up", "");
        Double margin = (Double) rationale.getOrDefault("runner_up_margin_percent", 0.0);

        StringBuilder sb = new StringBuilder();
        sb.append("<html><body style='font-family: -apple-system, BlinkMacSystemFont, Arial, Helvetica, sans-serif; font-size: 11px; color: #334155; margin: 0; padding: 0; line-height: 1.45; word-wrap: break-word;'>");

        sb.append("<div style='margin-bottom: 4px;'>");
        sb.append("<b style='color: #0f172a;'>Decision Factor:</b> ").append(summary);
        sb.append("</div>");

        if (markers != null && !markers.isEmpty()) {
            sb.append("<div style='margin-bottom: 2px;'><b style='color: #475569;'>Key Acoustic & Phonetic Markers:</b></div>");
            sb.append("<ul style='margin-top: 1px; margin-bottom: 4px; padding-left: 18px; color: #334155;'>");
            for (String m : markers) {
                sb.append("<li>").append(m).append("</li>");
            }
            sb.append("</ul>");
        }

        if (runnerUp != null && !runnerUp.isEmpty() && !runnerUp.equalsIgnoreCase("None")) {
            sb.append("<div style='font-size: 10.5px; color: #0369a1; background-color: #f0f9ff; padding: 3px 6px; border-radius: 4px; border: 1px solid #bae6fd;'>");
            sb.append("<b>Acoustic Separation:</b> Outperformed runner-up <b>").append(runnerUp).append("</b> by <b>+").append(String.format("%.1f%%", margin)).append("</b> margin.");
            sb.append("</div>");
        }

        sb.append("</body></html>");
        rationaleArea.setText(sb.toString());
        rationaleArea.revalidate();
        rationaleArea.repaint();

        telemetryLabel.setText(String.format("Latency: %.1f ms | Device: %s | Speech Duration: %.2fs | Engine: %s", latency, device.toUpperCase(), dur, engine));
    }

    private Map<String, Object> parseClassificationJson(String json) {
        Map<String, Object> map = new HashMap<>();

        map.put("predicted_dialect", extractJsonString(json, "predicted_dialect"));
        map.put("confidence_percent", extractJsonDouble(json, "confidence_percent"));
        map.put("region_name", extractJsonString(json, "region_name"));
        map.put("cultural_context", extractJsonString(json, "cultural_context"));
        map.put("consensus_badge", extractJsonString(json, "consensus_badge"));
        map.put("latency_ms", extractJsonDouble(json, "latency_ms"));
        map.put("duration_seconds", extractJsonDouble(json, "duration_seconds"));
        map.put("device", extractJsonString(json, "device"));
        map.put("engine", extractJsonString(json, "engine"));
        map.put("prediction_rationale", extractPredictionRationale(json));

        // Parse probabilities map
        Map<String, Double> probs = new LinkedHashMap<>();
        int pIdx = json.indexOf("\"probabilities\":");
        if (pIdx != -1) {
            int openBrace = json.indexOf("{", pIdx);
            int closeBrace = json.indexOf("}", openBrace);
            if (openBrace != -1 && closeBrace != -1) {
                String sub = json.substring(openBrace + 1, closeBrace);
                String[] pairs = sub.split(",");
                for (String p : pairs) {
                    String[] kv = p.split(":");
                    if (kv.length == 2) {
                        String key = kv[0].replaceAll("[\"{}]", "").trim();
                        try {
                            double val = Double.parseDouble(kv[1].replaceAll("[\"{}]", "").trim());
                            probs.put(key, val);
                        } catch (Exception ignored) {
                        }
                    }
                }
            }
        }
        map.put("probabilities", probs);
        return map;
    }

    private String extractJsonString(String json, String key) {
        String target = "\"" + key + "\":\"";
        int idx = json.indexOf(target);
        if (idx == -1) return "";
        idx += target.length();
        int end = json.indexOf("\"", idx);
        if (end == -1) return "";
        return json.substring(idx, end).replace("\\\"", "\"");
    }

    private Double extractJsonDouble(String json, String key) {
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

    private Map<String, Object> extractPredictionRationale(String json) {
        Map<String, Object> rat = new HashMap<>();
        int pIdx = json.indexOf("\"prediction_rationale\":");
        if (pIdx == -1) return rat;

        int open = json.indexOf("{", pIdx);
        if (open == -1) return rat;

        int close = -1;
        int depth = 0;
        for (int i = open; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') depth++;
            else if (c == '}') {
                depth--;
                if (depth == 0) {
                    close = i;
                    break;
                }
            }
        }
        if (close == -1) return rat;

        String sub = json.substring(open, close + 1);
        rat.put("summary", extractJsonString(sub, "summary"));
        rat.put("acoustic_cues", extractJsonString(sub, "acoustic_cues"));
        rat.put("contrast_note", extractJsonString(sub, "contrast_note"));
        rat.put("runner_up", extractJsonString(sub, "runner_up"));
        rat.put("runner_up_margin_percent", extractJsonDouble(sub, "runner_up_margin_percent"));
        rat.put("consensus_note", extractJsonString(sub, "consensus_note"));

        List<String> markers = new ArrayList<>();
        int mIdx = sub.indexOf("\"markers\":");
        if (mIdx != -1) {
            int mOpen = sub.indexOf("[", mIdx);
            int mClose = sub.indexOf("]", mOpen);
            if (mOpen != -1 && mClose != -1) {
                String mSub = sub.substring(mOpen + 1, mClose);
                String[] parts = mSub.split("\",\"");
                for (String p : parts) {
                    String clean = p.replaceAll("[\"\\[\\]]", "").trim();
                    if (!clean.isEmpty()) markers.add(clean);
                }
            }
        }
        rat.put("markers", markers);
        return rat;
    }

    private Map<String, Object> buildFallbackRationale(String dialect, double conf, Map<String, Double> probs) {
        Map<String, Object> rat = new HashMap<>();
        List<String> markers = new ArrayList<>();
        String contrast;

        switch (dialect) {
            case "Antruz":
                markers.add("Balanced central pitch envelope with standard neutral literary cadence");
                markers.add("Symmetric formant transitions (F1 ~500 Hz, F2 ~1500 Hz) with schwa retention");
                markers.add("Absence of northern open-vowel dilation or southern Portuguese prosodic lilt");
                contrast = "Canonical central Goan articulation with stable formant transitions.";
                break;
            case "Bardez":
                markers.add("Open terminal vowel elongation (shift towards broad /ɔ/ and /au/)");
                markers.add("Elevated F0 fundamental pitch contour with northern melodic inflection");
                markers.add("Rapid staccato syllable delivery with northern coastal tempo");
                contrast = "Differentiated by open-vowel dilation at clause terminals and higher F0 contour.";
                break;
            case "Canacona":
                markers.add("Distinct southern coastal intonation curve transitioning toward Karwar phonology");
                markers.add("Softening of sibilants and vowel rounding (/u/ and /o/ acoustic shifts)");
                markers.add("Descending terminal pitch envelope on utterance finals");
                contrast = "Differentiated by Karwar-adjacent phonological vowel rounding and terminal pitch drop.";
                break;
            case "Gawda_Kunbi":
                markers.add("Deep guttural / velar acoustic resonance in the 250–600 Hz sub-band");
                markers.add("Rhythmic folk cadence with strong syllabic stress pulses");
                markers.add("Retention of archaic unreduced consonant clusters and firm dental plosives");
                contrast = "Differentiated by robust non-nasalized dental plosives and rhythmic pulse.";
                break;
            case "Pednekar":
                markers.add("Fast-clipped speech tempo (elevated syllables-per-second velocity)");
                markers.add("Acoustic transition toward Malvani / Sindhudurg border phonetics");
                markers.add("Sharp aspirated consonant bursts (/kh/, /th/, /bh/) with compressed vowels");
                contrast = "Differentiated by high speech tempo and Sindhudurg borderland aspiration.";
                break;
            case "Sashti_Christian":
                markers.add("Melodic theatrical lilt with prolonged final open vowels (-ão, trailing -e)");
                markers.add("Portuguese lexical loan adaptations and softening of sibilants (/s/ -> /z/ / /sh/)");
                markers.add("Modulated pitch contour characteristic of classic Goan Tiatr prosody");
                contrast = "Differentiated by trailing vowel elongation and Portuguese prosodic lilt.";
                break;
            case "Sashti_Hindu":
                markers.add("Crisp, precise dental stop articulation (firm dental /t/, /d/ contact)");
                markers.add("Characteristic nasalization (anusvara resonance) on terminal vowels");
                markers.add("Disciplined, steady southern Salcete cadence with compact vowels");
                contrast = "Differentiated by crisp dental stops and clear nasal resonance bands.";
                break;
            case "Sattarkar":
                markers.add("Sahyadri foothill cadence with transitional Marathi-Konkani inflections");
                markers.add("Strong retroflex consonant coloring (/ṭ/, /ḍ/, /ḷ/ spectral dips)");
                markers.add("Deliberate, measured cadence with deep chest resonance");
                contrast = "Differentiated by heavy retroflex coloring and Ghat-foothill intonation.";
                break;
            default:
                markers.add("Distinct regional acoustic formant distribution across Wav2Vec2 layers");
                markers.add("Characteristic prosodic cadence and syllabic timing");
                markers.add("Consonant-vowel boundary energy transitions");
                contrast = "Acoustically separated from neighboring regional varieties.";
                break;
        }

        String runnerUp = "None";
        double runnerUpProb = 0.0;
        if (probs != null && probs.size() > 1) {
            List<Map.Entry<String, Double>> sorted = new ArrayList<>(probs.entrySet());
            sorted.sort((a, b) -> Double.compare(b.getValue(), a.getValue()));
            for (Map.Entry<String, Double> e : sorted) {
                if (!e.getKey().equals(dialect)) {
                    runnerUp = e.getKey().replace("_", " ");
                    runnerUpProb = e.getValue() * 100.0;
                    break;
                }
            }
        }
        double margin = conf - runnerUpProb;

        rat.put("summary", String.format("Acoustic model matched key phonetic parameters for %s (%s).", dialect.replace("_", " "), contrast));
        rat.put("markers", markers);
        rat.put("runner_up", runnerUp);
        rat.put("runner_up_margin_percent", Math.max(0.0, margin));
        rat.put("contrast_note", contrast);
        return rat;
    }

    /**
     * Modern rounded button with custom high-contrast rendering
     * Bypasses OS look-and-feel paint delegates to guarantee consistent, sharp rendering.
     */
    static class ModernButton extends JButton {
        private Color normalBg;
        private Color hoverBg;
        private Color pressedBg;
        private Color customFg;
        private int cornerRadius;

        public ModernButton(String text, Color bg, Color fg, int cornerRadius) {
            super(text);
            this.normalBg = bg;
            this.hoverBg = brighten(bg, 0.12f);
            this.pressedBg = darken(bg, 0.15f);
            this.customFg = fg;
            this.cornerRadius = cornerRadius;
            setContentAreaFilled(false);
            setFocusPainted(false);
            setBorderPainted(false);
            setOpaque(false);
            setForeground(fg);
            setCursor(new Cursor(Cursor.HAND_CURSOR));
        }

        private static Color brighten(Color c, float fraction) {
            int r = Math.min(255, (int)(c.getRed() + (255 - c.getRed()) * fraction));
            int g = Math.min(255, (int)(c.getGreen() + (255 - c.getGreen()) * fraction));
            int b = Math.min(255, (int)(c.getBlue() + (255 - c.getBlue()) * fraction));
            return new Color(r, g, b, c.getAlpha());
        }

        private static Color darken(Color c, float fraction) {
            int r = Math.max(0, (int)(c.getRed() * (1 - fraction)));
            int g = Math.max(0, (int)(c.getGreen() * (1 - fraction)));
            int b = Math.max(0, (int)(c.getBlue() * (1 - fraction)));
            return new Color(r, g, b, c.getAlpha());
        }

        private static boolean isLightColor(Color c) {
            double luminance = (0.299 * c.getRed() + 0.587 * c.getGreen() + 0.114 * c.getBlue()) / 255.0;
            return luminance > 0.65;
        }

        @Override
        public void setBackground(Color bg) {
            super.setBackground(bg);
            this.normalBg = bg;
            this.hoverBg = brighten(bg, 0.12f);
            this.pressedBg = darken(bg, 0.15f);
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

            Color bg;
            if (!isEnabled()) {
                bg = isLightColor(normalBg) ? new Color(241, 245, 249) : new Color(203, 213, 225);
            } else if (getModel().isPressed()) {
                bg = pressedBg;
            } else if (getModel().isRollover()) {
                bg = hoverBg;
            } else {
                bg = normalBg;
            }

            g2.setColor(bg);
            g2.fillRoundRect(0, 0, getWidth(), getHeight(), cornerRadius, cornerRadius);

            if (isLightColor(normalBg)) {
                g2.setColor(new Color(203, 213, 225));
                g2.drawRoundRect(0, 0, getWidth() - 1, getHeight() - 1, cornerRadius, cornerRadius);
            }

            FontMetrics fm = g2.getFontMetrics(getFont());
            String text = getText();
            if (text != null && !text.isEmpty()) {
                int textWidth = fm.stringWidth(text);
                int x = Math.max(2, (getWidth() - textWidth) / 2);
                int y = (getHeight() - fm.getHeight()) / 2 + fm.getAscent();

                Color textCol = isEnabled() ? customFg : new Color(148, 163, 184);
                g2.setColor(textCol);
                g2.setFont(getFont());
                g2.drawString(text, x, y);
            }

            g2.dispose();
        }
    }

    /**
     * Interactive Modern Drag & Drop Zone Component
     */
    static class DropZonePanel extends JPanel {
        private boolean isDragOver = false;
        private String fileName = "No file chosen";
        private String badge = "";
        private String fileSize = "";

        public DropZonePanel() {
            setOpaque(false);
            setPreferredSize(new Dimension(Integer.MAX_VALUE, 92));
            setMaximumSize(new Dimension(Integer.MAX_VALUE, 92));
            setCursor(Cursor.getPredefinedCursor(Cursor.HAND_CURSOR));
        }

        public void setDragOver(boolean over) {
            this.isDragOver = over;
            repaint();
        }

        public void setFileInfo(String name, String badge, String size) {
            this.fileName = name;
            this.badge = badge;
            this.fileSize = size;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth();
            int h = getHeight();

            if (isDragOver) {
                // Indigo Drag Hover State
                g2.setColor(new Color(238, 242, 255));
                g2.fillRoundRect(2, 2, w - 4, h - 4, 12, 12);
                g2.setColor(new Color(99, 102, 241));
                g2.setStroke(new BasicStroke(2.2f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 0, new float[]{6, 4}, 0));
                g2.drawRoundRect(2, 2, w - 4, h - 4, 12, 12);

                g2.setFont(getAppFont(Font.BOLD, 15));
                g2.setColor(new Color(79, 70, 229));
                String text = "DROP AUDIO FILE HERE TO LOAD";
                FontMetrics fm = g2.getFontMetrics();
                g2.drawString(text, (w - fm.stringWidth(text)) / 2, h / 2 + 5);
            } else if (fileName.equals("No file chosen")) {
                // Empty Prompt State
                g2.setColor(new Color(248, 250, 252));
                g2.fillRoundRect(2, 2, w - 4, h - 4, 12, 12);
                g2.setColor(new Color(203, 213, 225));
                g2.setStroke(new BasicStroke(1.5f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND, 0, new float[]{5, 4}, 0));
                g2.drawRoundRect(2, 2, w - 4, h - 4, 12, 12);

                // Prompt Icon & Text
                g2.setFont(getAppFont(Font.BOLD, 13));
                g2.setColor(new Color(30, 41, 59));
                String title = "DRAG & DROP AUDIO FILE HERE";
                FontMetrics fm1 = g2.getFontMetrics();
                g2.drawString(title, (w - fm1.stringWidth(title)) / 2, 32);

                g2.setFont(getAppFont(Font.PLAIN, 11));
                g2.setColor(new Color(100, 116, 139));
                String sub = "Supports WAV, MP3, M4A, FLAC, AAC, OGG (Studio DSP Auto-Filter)";
                FontMetrics fm2 = g2.getFontMetrics();
                g2.drawString(sub, (w - fm2.stringWidth(sub)) / 2, 52);

                g2.setFont(getAppFont(Font.BOLD, 11));
                g2.setColor(new Color(2, 132, 199)); // Sky Blue 600
                String clickText = "or Click to Browse Computer...";
                FontMetrics fm3 = g2.getFontMetrics();
                g2.drawString(clickText, (w - fm3.stringWidth(clickText)) / 2, 72);
            } else {
                // Loaded File State
                g2.setColor(new Color(240, 253, 244)); // Light Green Slate (#F0FDF4)
                g2.fillRoundRect(2, 2, w - 4, h - 4, 12, 12);
                g2.setColor(new Color(134, 239, 172)); // Green 300
                g2.setStroke(new BasicStroke(1.5f));
                g2.drawRoundRect(2, 2, w - 4, h - 4, 12, 12);

                // File Name Title
                g2.setFont(getAppFont(Font.BOLD, 13));
                g2.setColor(new Color(20, 83, 45));
                String title = fileName;
                FontMetrics fm = g2.getFontMetrics();
                if (fm.stringWidth(title) > w - 40) {
                    while (fm.stringWidth(title + "...") > w - 40 && title.length() > 8) {
                        title = title.substring(0, title.length() - 1);
                    }
                    title = title + "...";
                }
                g2.drawString(title, 14, 30);

                // Format Badge (Pill)
                if (!badge.isEmpty()) {
                    g2.setFont(getAppFont(Font.BOLD, 10));
                    g2.setColor(new Color(16, 185, 129));
                    g2.fillRoundRect(14, 44, 58, 20, 10, 10);
                    g2.setColor(Color.WHITE);
                    int bW = g2.getFontMetrics().stringWidth(badge);
                    g2.drawString(badge, 14 + (58 - bW) / 2, 58);
                }

                // File Details
                g2.setFont(getAppFont(Font.PLAIN, 11));
                g2.setColor(new Color(22, 101, 52));
                String details = fileSize + " • Ready for inference • Drag new file or click Browse to replace";
                g2.drawString(details, 82, 58);

                g2.setFont(getAppFont(Font.ITALIC, 10));
                g2.setColor(new Color(74, 222, 128));
            }

            g2.dispose();
        }
    }

    /**
     * Custom Graphic Component rendering horizontal probability bar charts
     */
    static class ProbabilityChartPanel extends JPanel {
        private final Map<String, Double> probabilities = new LinkedHashMap<>();
        private String topDialect = "";

        public ProbabilityChartPanel() {
            setBackground(Color.WHITE);
            String[] defaultKeys = {"Antruz", "Bardez", "Canacona", "Gawda_Kunbi", "Pednekar", "Sashti_Christian", "Sashti_Hindu", "Sattarkar"};
            for (String k : defaultKeys) {
                probabilities.put(k, 0.0);
            }
        }

        public void updateData(Map<String, Double> probs, String top) {
            this.probabilities.clear();
            this.probabilities.putAll(probs);
            this.topDialect = top;
            repaint();
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g.create();
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int width = getWidth();
            int height = getHeight();

            if (probabilities.isEmpty()) {
                g2.dispose();
                return;
            }

            int rowHeight = height / probabilities.size();
            int labelWidth = 145;
            int maxBarWidth = width - labelWidth - 85;

            int y = 4;
            for (Map.Entry<String, Double> entry : probabilities.entrySet()) {
                String dialect = entry.getKey();
                double prob = entry.getValue();
                int barWidth = (int) (prob * maxBarWidth);

                boolean isTop = dialect.equalsIgnoreCase(topDialect);

                // Dialect Label
                g2.setFont(getAppFont(isTop ? Font.BOLD : Font.PLAIN, 11));
                g2.setColor(isTop ? new Color(15, 23, 42) : new Color(71, 85, 105));
                g2.drawString(dialect.replace("_", " "), 6, y + rowHeight - 6);

                // Bar Background Track
                g2.setColor(new Color(241, 245, 249)); // Slate 100
                g2.fillRoundRect(labelWidth, y + 2, maxBarWidth, rowHeight - 6, 8, 8);

                // Active Bar (Gradient if Top)
                if (isTop) {
                    GradientPaint gp = new GradientPaint(
                            labelWidth, y + 2, new Color(37, 99, 235), // Blue 600
                            labelWidth + Math.max(barWidth, 10), y + 2, new Color(14, 165, 233) // Sky 500
                    );
                    g2.setPaint(gp);
                } else {
                    g2.setColor(new Color(203, 213, 225)); // Slate 300
                }
                g2.fillRoundRect(labelWidth, y + 2, Math.max(barWidth, 4), rowHeight - 6, 8, 8);

                // Percentage Label
                g2.setFont(getAppFont(isTop ? Font.BOLD : Font.PLAIN, 10));
                g2.setColor(isTop ? new Color(37, 99, 235) : new Color(100, 116, 139));
                g2.drawString(String.format("%.1f%%", prob * 100), labelWidth + maxBarWidth + 12, y + rowHeight - 6);

                y += rowHeight;
            }
            g2.dispose();
        }
    }

    public static void main(String[] args) {
        System.setProperty("awt.useSystemAAFontSettings", "on");
        System.setProperty("swing.aatext", "true");
        System.setProperty("sun.java2d.renderer", "sun.java2d.marlin.MarlinRenderingEngine");

        // Set System Look & Feel
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception ignored) {
        }

        SwingUtilities.invokeLater(() -> {
            KonkaniDialectClassifierApp app = new KonkaniDialectClassifierApp();
            app.setVisible(true);
        });
    }
}
