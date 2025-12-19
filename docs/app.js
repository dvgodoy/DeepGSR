// Import Transformers.js from CDN
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

// 1. Disable remote downloads to stop the "Invalid Model ID" fallback error
env.allowRemoteModels = false;
env.allowLocalModels = true;
//env.useBrowserCache = false;

// 2. Set the base path for your models. 
// If your 'models' folder is in the same directory as index.html, use './'
env.localModelPath = './';

const classifierCache = {};
const config = {
    PAS: { length: 606, motif: "AATAAA", motifStart: 300, motifEnd: 306, path: 'models/PAS_model' },
    TIS: { length: 603, motif: "ATG",    motifStart: 300, motifEnd: 303, path: 'models/TIS_model' }
};

const btn = document.getElementById('classifyBtn');
const resultDiv = document.getElementById('result');
const statusDiv = document.getElementById('status');

async function checkWebGPUSupport() {
    if (!navigator.gpu) return false;
    try {
        const adapter = await navigator.gpu.requestAdapter();
        return !!adapter;
    } catch (e) {
        return false;
    }
}

async function classify() {
    const type = document.getElementById('signalType').value;
    const rawSequence = document.getElementById('sequenceInput').value.trim().toUpperCase();

    const settings = config[type];

    // 1. Basic Length and Character Validation
    if (rawSequence.length !== settings.length) {
        showResult(`Invalid length. ${type} requires ${settings.length} nts.`, true);
        return;
    }
    if (!/^[ACGT]+$/.test(rawSequence)) {
        showResult("Sequence must only contain A, C, G, and T.", true);
        return;
    }

    // 2. Motif Validation (Positions 301+)
    const actualMotif = rawSequence.substring(settings.motifStart, settings.motifEnd);
    if (actualMotif !== settings.motif) {
        showResult(`Invalid Signal: Expected ${settings.motif} at position 301, found ${actualMotif}.`, true);
        return;
    }

    // 3. Upstream and Downstream Extraction
    // Upstream: 1 to 300 (Indices 0 to 300)
    // Downstream: (Following motif) to End
    let upstream = rawSequence.substring(0, 300);
    let downstream = rawSequence.substring(settings.motifEnd);

    // 4. Trimming: Remove the first 3 nucleotides from each
    upstream = upstream.substring(3);
    downstream = downstream.substring(3);

    // 5. Final Concatenation
    const processedSequence = upstream + downstream;
    // Length check: (300-3) + (300-3) = 594 nts (198 tokens)

    try {
        btn.disabled = true;
        statusDiv.style.display = 'block';
        resultDiv.style.display = 'none';

        if (!classifierCache[type]) {
            statusDiv.innerText = `Loading ${type} model...`;
            classifierCache[type] = await pipeline('text-classification', settings.path, { 
                quantized: false, dtype: 'fp32'
            });
        }

        statusDiv.innerText = "Analyzing processed sequence...";
        const output = await classifierCache[type](processedSequence);

        const { label, score } = output[0];
        const isPositive = label === 'LABEL_1' || label.toUpperCase() === 'POSITIVE';

        showResult(`
            <strong>Result:</strong> ${isPositive ? '✅ Positive' : '❌ Negative'}<br>
            <strong>Confidence:</strong> ${(score * 100).toFixed(2)}%<br>
        `, false);

    } catch (err) {
        console.error(err);
        showResult("Error during classification. Ensure model files are in the correct path.", true);
    } finally {
        btn.disabled = false;
        statusDiv.style.display = 'none';
    }
}

function showResult(html, isError) {
    resultDiv.innerHTML = html;
    resultDiv.style.display = 'block';
    resultDiv.className = isError ? 'error' : 'success';
}
btn.addEventListener('click', classify);    

let batchResults = []; // Store results for CSV

// --- 1. Simple FASTA Parser ---
function parseFASTA(text) {
    const sequences = [];
    const lines = text.split('\n');
    let currentId = '';
    let currentSeq = '';

    for (let line of lines) {
        line = line.trim();
        if (line.startsWith('>')) {
            if (currentId) sequences.push({ id: currentId, seq: currentSeq });
            currentId = line.substring(1);
            currentSeq = '';
        } else {
            currentSeq += line;
        }
    }
    if (currentId) sequences.push({ id: currentId, seq: currentSeq });
    return sequences;
}

// Helper to unblock the UI thread
const yieldToBrowser = () => new Promise(resolve => setTimeout(resolve, 0));

let worker = null;

function initWorker() {
    if (worker) worker.terminate(); // Kill any existing worker
    
    worker = new Worker('https://dvgodoy.github.io/DeepGSR/worker.js', { type: 'module' });
    //worker = new Worker('worker.js');

    worker.onmessage = (e) => {
        const { status, message, current, total, results } = e.data;
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const downloadBtn = document.getElementById('downloadBtn');
        const stopBtn = document.getElementById('stopBatchBtn');
        const badge = document.getElementById('hardwareBadge');
        
        if (status === 'engine-ready') {
            badge.classList.remove('badge-hidden', 'badge-gpu', 'badge-cpu');
            
            if (e.data.device === 'webgpu') {
                badge.innerText = "🚀 Hardware: GPU";
                badge.classList.add('badge-gpu');
            } else {
                badge.innerText = "💻 Hardware: CPU";
                badge.classList.add('badge-cpu');
            }
        }

        if (status === 'loading') {
            progressText.innerText = message;
            badge.classList.remove('badge-hidden', 'badge-gpu', 'badge-cpu');
            badge.innerText = "Detecting Hardware...";
        } else if (status === 'progress') {
            progressBar.value = (current / total) * 100;
            progressText.innerText = `Processed ${current} / ${total} sequences...`;
        } else if (status === 'complete') {
            batchResults = results;
            progressText.innerText = "Batch processing complete!";
            downloadBtn.style.display = 'block';
            stopBtn.style.display = 'none'; // Hide stop once finished
        } else if (status === 'error') {
            alert("Worker Error: " + message);
            resetBatchUI();
        }
    };
}

document.getElementById('stopBatchBtn').addEventListener('click', () => {
    if (worker) {
        worker.terminate();
        worker = null; // Clear the reference
        console.log("Worker terminated by user.");
    }
    
    // Reset UI
    const progressText = document.getElementById('progressText');
    progressText.innerText = "Process stopped by user.";
    progressText.style.color = "#dc3545";
    
    // Hide the stop button but keep the container visible for a moment
    document.getElementById('stopBatchBtn').style.display = 'none';
    
    setTimeout(() => {
        document.getElementById('progressContainer').style.display = 'none';
        progressText.style.color = ""; // Reset color
    }, 2000);
});

async function handleBatch() {
    const fileInput = document.getElementById('fastaInput');
    if (!fileInput.files.length) return alert("Select a file.");

    const type = document.getElementById('signalType').value;
    const text = await fileInput.files[0].text();
    const sequences = parseFASTA(text); // Use your existing parser

    const hasWebGPU = await checkWebGPUSupport();
    console.log(hasWebGPU ? "WebGPU detected. Using GPU acceleration." : "WebGPU not detected. Falling back to CPU.");

    // Update UI to let user know
    const progressText = document.getElementById('progressText');
    progressText.innerText = hasWebGPU ? "Initializing GPU Engine..." : "Initializing CPU Engine...";

    initWorker(); // Create a fresh worker for this batch
    
    document.getElementById('progressContainer').style.display = 'block';
    document.getElementById('stopBatchBtn').style.display = 'block';
    document.getElementById('downloadBtn').style.display = 'none';

    worker.postMessage({
        type: type,
        sequences: sequences,
        settings: config[type],
        //batchSize: hasWebGPU ? 64 : 16, // We can use LARGER batches on GPU!
        batchSize: 64,
        useGPU: hasWebGPU
    });
}

function resetBatchUI() {
    // Clear internal data to free up memory
    batchResults = [];

    // Reset the File Input (crucial so the same file can be re-selected if needed)
    const fileInput = document.getElementById('fastaInput');
    fileInput.value = ''; 

    // Reset and hide the Progress Bar
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const progressContainer = document.getElementById('progressContainer');
    
    progressBar.value = 0;
    progressText.innerText = 'Ready for next batch...';
    
    // Hide the download button and progress container after a short delay
    // (A 2-second delay lets the user see the "complete" state before it vanishes)
    setTimeout(() => {
        progressContainer.style.display = 'none';
        document.getElementById('downloadBtn').style.display = 'none';
    }, 2000);
}    

function downloadCSV() {
    if (batchResults.length === 0) return;

    // Add "GC Content (%)" to the header
    let csvContent = "data:text/csv;charset=utf-8,ID,Status,Classification,Probability,GC Content (%)\n";
    
    batchResults.forEach(r => {
        // Append the GC content to each row
        csvContent += `${r.id},${r.status},${r.label},${r.probability},${r.gcContent}\n`;
    });

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `genomic_results_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    resetBatchUI();
}

function updateStats() {
    const inputArea = document.getElementById('sequenceInput');
    const type = document.getElementById('signalType').value;
    const targetLength = config[type].length;

    // Remove any non-DNA characters (whitespace, numbers, etc.) for the calculation
    const cleanSeq = inputArea.value.toUpperCase().replace(/[^ACGTN]/g, "");
    
    const len = cleanSeq.length;
    
    // 1. Update Length Display
    const lengthSpan = document.getElementById('statLength');
    lengthSpan.innerText = len;

    // 2. Visual Feedback: Green if length matches the model requirements, Red otherwise
    if (len === targetLength) {
        lengthSpan.style.color = "#28a745"; // Success Green
        lengthSpan.style.fontWeight = "bold";
    } else {
        lengthSpan.style.color = "#dc3545"; // Danger Red
        lengthSpan.style.fontWeight = "normal";
    }

    // 3. Update GC-Content Display
    let gcPercent = 0;
    if (len > 0) {
        const gcCount = (cleanSeq.match(/[GC]/g) || []).length;
        gcPercent = (gcCount / len) * 100;
    }
    document.getElementById('statGC').innerText = gcPercent.toFixed(2);
}

// --- Sample Data Generation ---
const samples = {
    PAS: {
        motif: "AATAAA",
        length: 606,
        // Helper to build a valid string: 300 random + AATAAA + 300 random
        generate: () => {
            //const pad = (n) => Array.from({length: n}, () => "ACGT"[Math.floor(Math.random() * 4)]).join("");
            //return pad(300) + "AATAAA" + pad(300);
            return "AGGAGGAGAAGAAGGAGGAGGATGGATCCCCTGATGCCTTTCCTCCATCCCTGTCTCTCCCCCAGACTGATTCTTCCAGACCAGAGTTTGATGCCAGCAGCTTCGGCCATCCAAACAGAGGATGCTCAGATTTCTCACATCCTGCCCAGGATCTCCTCTTAGGGTAGAAGAAGTCTCTGGGACATCCCTGGGGTGTGTGTGTAGATTTCCCACCTGGGGACTCTGCTGTCCCTGGGCTTGCATCCCAGGGATCCCAGAGTGGCCTGCCTATCACAACCACATCCCTTCCCCCCACAAGGCAATAAATCTCATTTCTTTATATCAGTGTGGCTTCTTTCTTAACTCATGGTATTTGTTTCTGGATATCTCAACTTGAGTGGGTTGTCGTTTCAAATTCAGCATGCCTTAACCTGAACACAGCTTGACCTCGTTAGGGAGGGAAATAGGGAAAACCCCTAATTTGCCAGCTGAGCTCTTATTCCCTGGTCTTGGCGGTACATGATGTTTTTCCATCTATCGGTTTGTGCAAAATATGTGAGAAACGAAGGCAGAGTTATTTTCTAATAATCTGCTTACAAAATGGTTAAGGAAGCTGCTTGTGTGTTT";
        }
    },
    TIS: {
        motif: "ATG",
        length: 603,
        // Helper: 300 random + ATG + 300 random
        generate: () => {
            //const pad = (n) => Array.from({length: n}, () => "ACGT"[Math.floor(Math.random() * 4)]).join("");
            //return pad(300) + "ATG" + pad(300);
            return "CGGCGGCGGGCGGCGCGACGAGCCCCTGTGATTGGCACAGCCGGAGCCGGAGGAGGAGGCCAGGGGAGGGCGGAGGCGGGGGAGGAGGAGGAGGAAGGGGCGATCGCGGCGGCGGCGGCGGCGGCGAGGAGCTGTGCCTTCCACCTCTCCAGCCCCGGCAGGACGGGGGCGGCCGCCGCGAACCCGGGGCGGGGACAGCACGCAGCCTCGAGGCGCGCACCCCCGCCCGGCAGCGGCCCCGACACCCGGGGCGAGCGGGAAAGCGGCAGCGGCGGCGGCGGCGGCGGCGGCGGGGGAAGGATGCAGGGGAAGAAGCCGGGCGGTTCGTCGGGCGGCGGCCGGAGCGGCGAGCTGCAGGGGGACGAGGCGCAGAGGAACAAGAAAAAGAAAAAGAAGGTGTCCTGCTTTTCCAACATCAAGATCTTCCTGGTGTCCGAGTGCGCCCTGATGCTGGCGCAGGGCACGGTGGGCGCCTACCTGGTGAGTCCCCGAGCCAACTCCGCCGCGGGCCCCTTCCCCAGCCCGGCTCTCGAGCGGCCGCCTGGCCCGACGAGGGGGCCGCCCGGCGCTGGGGGCAGGCGGGCATGACCTCGGCCCGGCGTG";
        }
    }
};

document.getElementById('clearSingleBtn').addEventListener('click', () => {
    // 1. Clear the text
    const inputArea = document.getElementById('sequenceInput');
    inputArea.value = '';
    
    // 2. Hide the previous result
    const resultDiv = document.getElementById('result');
    resultDiv.style.display = 'none';
    resultDiv.innerHTML = '';

    // 3. Update the stats (Length and GC) back to 0
    updateStats(); 
});

const inputArea = document.getElementById('sequenceInput');
const dropdown = document.getElementById('signalType');

inputArea.addEventListener('input', updateStats);

// Also update when the signal type changes (to update the red/green length color)
document.getElementById('signalType').addEventListener('change', updateStats);

// --- Event Listeners for Sample Buttons ---
document.getElementById('samplePasBtn').addEventListener('click', () => {
    inputArea.value = samples.PAS.generate();
    dropdown.value = "PAS";
    updateStats();
    showResult("Sample PAS loaded. Note the AATAAA at position 301.", false);
});

document.getElementById('sampleTisBtn').addEventListener('click', () => {
    inputArea.value = samples.TIS.generate();
    dropdown.value = "TIS";
    updateStats();
    showResult("Sample TIS loaded. Note the ATG at position 301.", false);
});

// Add Event Listeners
document.getElementById('batchProcessBtn').addEventListener('click', handleBatch);
document.getElementById('downloadBtn').addEventListener('click', downloadCSV);

document.getElementById('resetBtn').addEventListener('click', () => {
    // Clear everything
    document.getElementById('sequenceInput').value = '';
    document.getElementById('fastaInput').value = '';
    document.getElementById('result').style.display = 'none';
    document.getElementById('progressContainer').style.display = 'none';
    updateStats();
});