import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers';

// Use the bundled version from the CDN that attaches to 'self'
//importScripts('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0/dist/transformers.min.js');
// Access the library via the global 'self.transformers' object
//const { pipeline, env } = self.transformers;

// Configure environment inside the worker
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = './'; 

let classifier = null;
let currentType = null;

async function loadModel(type, path, useGPU) {
    try {
        const device = useGPU ? 'webgpu' : 'wasm';
        const dtype = useGPU ? 'fp16' : 'fp32';
        
        self.postMessage({ 
            status: 'loading', 
            message: `Initializing ${type} model on ${device.toUpperCase()}...` 
        });

        classifier = await pipeline('text-classification', path, { 
            device: device,
            dtype: dtype,
            // We only force the quantized file if we are on the CPU
            quantized: !useGPU, 
            // This tells Transformers.js to look for 'model_quantized.onnx'
            model_file_name: useGPU ? 'model' : 'model_quantized' 
        });
        
        currentType = type;
        self.postMessage({ status: 'engine-ready', device: device });
        return true;
    } catch (err) {
        if (useGPU) {
            console.warn("WebGPU failed to initialize. Falling back to CPU...", err);
            // RECURSION: Try again, but force useGPU to false
            return await loadModel(type, path, false);
        } else {
            // If even the CPU fails, we have a real problem
            throw err;
        }
    }
}

// The Worker listens for messages from the Main Thread
self.onmessage = async (e) => {
    const { type, sequences, settings, batchSize, useGPU } = e.data;

    try {
        // Only load if the model changed or hasn't been loaded yet
        if (!classifier || currentType !== type) {
            await loadModel(type, settings.path, useGPU);
        }

        const results = [];
        
        // 2. Process in Batches
        for (let i = 0; i < sequences.length; i += batchSize) {
            const chunk = sequences.slice(i, i + batchSize);
            const validSeqs = [];
            const meta = [];

            chunk.forEach((item, idx) => {
                const raw = item.seq.toUpperCase();
                const gc = (((raw.match(/[GC]/g) || []).length / raw.length) * 100).toFixed(2);
                
                const isValid = raw.length === settings.length && 
                                raw.substring(settings.motifStart, settings.motifEnd) === settings.motif &&
                                !(!/^[ACGT]+$/.test(raw));

                if (isValid) {
                    let upstream = raw.substring(0, 300).substring(3);
                    let downstream = raw.substring(settings.motifEnd).substring(3);
                    validSeqs.push(upstream + downstream);
                    meta.push({ id: item.id, valid: true, gcContent: gc });
                } else {
                    meta.push({ id: item.id, valid: false, gcContent: gc });
                }
            });

            // Run Inference
            let modelOutputs = [];
            if (validSeqs.length > 0) {
                modelOutputs = await classifier(validSeqs);
            }

            // Map back
            let outputIdx = 0;
            meta.forEach(m => {
                if (m.valid) {
                    const out = modelOutputs[outputIdx++];
                    const isPositive = out.label === 'LABEL_1' || out.label.toUpperCase() === 'POSITIVE';
                    results.push({
                        id: m.id, status: 'Success', 
                        label: isPositive ? 'Positive' : 'Negative',
                        probability: out.score.toFixed(4), gcContent: m.gcContent
                    });
                } else {
                    results.push({
                        id: m.id, status: 'Invalid (Sequence/Length/Motif)', 
                        label: 'N/A', probability: '0.0000', gcContent: m.gcContent
                    });
                }
            });

            // 3. Send progress update back to UI
            self.postMessage({ 
                status: 'progress', 
                current: Math.min(i + batchSize, sequences.length), 
                total: sequences.length 
            });
        }

        // 4. Send final results
        self.postMessage({ status: 'complete', results });

    } catch (finalError) {
        self.postMessage({ status: 'error', message: "Critical Error: " + finalError.message });
    }
};