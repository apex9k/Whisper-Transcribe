import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js to use the extension's storage
env.localModelPath = chrome.runtime.getURL('models/');
env.allowRemoteModels = true;  // Allow downloading models from HuggingFace
env.useBrowserCache = true;    // Enable browser cache for model files
env.cacheDir = 'transformers-cache'; // Set a specific cache directory name

// Configure ONNX runtime settings
env.backends.onnx.wasm.numThreads = 1;  // Use single thread for better compatibility
env.backends.onnx.wasm.simd = false;    // Disable SIMD for better compatibility

// Log the model path for debugging
console.log('Model path:', env.localModelPath);
console.log('ONNX configuration:', env.backends.onnx.wasm);

// State variables
let transcriber = null;
let processor = null;
let model = null;
let recording = false;
let mediaRecorder = null;
let audioChunks = [];
let selectedModel = 'Xenova/whisper-tiny.en';
let recordingTimer = null;
let recordingStartTime = 0;
let loadingProgress = 0;
let isWhisperModel = true;

// Add visualizer variables
let audioContext;
let analyser;
let dataArray;
let canvas;
let canvasCtx;
let animationId;
let visualizerInitialized = false;

// DOM elements
const recordBtn = document.getElementById('record-btn');
const transcriptionEl = document.getElementById('transcription');
const copyBtn = document.getElementById('copy-btn');
const clearBtn = document.getElementById('clear-btn');
const modelSelect = document.getElementById('model-select');
const statusEl = document.getElementById('status');
const timerEl = document.getElementById('timer');
const modelInfoEl = document.getElementById('model-info');
const progressBar = document.getElementById('progress-bar');

// Clean up Whisper output by removing special tags
function cleanWhisperOutput(text) {
  if (!text) return '';
  
  // Remove Whisper special tags
  return text
    .replace(/<\|startoftranscript\|>/g, '')
    .replace(/<\|endoftranscript\|>/g, '')
    .replace(/<\|endoftext\|>/g, '')
    .replace(/<\|notimestamps\|>/g, '')
    .replace(/<\|transcribe\|>/g, '')
    .replace(/<\|translate\|>/g, '')
    // Additional tags that might appear in some models
    .replace(/\[START\]/g, '')
    .replace(/\[END\]/g, '')
    .replace(/\[\w+\]/g, '') // Remove any bracketed tags
    .replace(/\s+/g, ' ') // Normalize whitespace
    .replace(/^\s*\.\s*$/, '') // Remove lone periods
    .replace(/^[.,;:!?]+/, '') // Remove leading punctuation
    .replace(/[.,;:!?]+$/, '') // Remove trailing punctuation
    .trim();
}

// Progress callback
const progressCallback = (progress) => {
  if (progress.status === 'download') {
    loadingProgress = progress.progress || 0;
    progressBar.style.width = `${loadingProgress * 100}%`;
    statusEl.textContent = `Downloading model: ${Math.round(loadingProgress * 100)}%`;
  } else if (progress.status === 'init') {
    statusEl.textContent = 'Initializing model...';
  }
};

// Initialize the model
async function initializeModel() {
  try {
    statusEl.textContent = 'Loading model...';
    progressBar.style.width = '0%';
    
    console.log('Starting model initialization for:', selectedModel);
    isWhisperModel = selectedModel.toLowerCase().includes('whisper');
    
    // First try to ping the background service worker to ensure it's active
    try {
      await chrome.runtime.sendMessage({ action: 'ping' });
      console.log('Background service worker is active');
    } catch (error) {
      console.warn('Background service worker not active:', error);
      // Don't fail here, continue with local loading
    }
    
    // Try to use the transcriber from the background script first
    try {
      const response = await new Promise((resolve, reject) => {
        chrome.runtime.sendMessage({ 
          action: 'preloadModel', 
          model: selectedModel 
        }, (response) => {
          if (chrome.runtime.lastError) {
            reject(chrome.runtime.lastError);
          } else {
            resolve(response);
          }
        });
      });
      
      console.log('Model preloaded by background script:', response);
    } catch (error) {
      console.warn('Failed to preload model in background:', error);
      // Don't fail here, continue with local loading
    }
    
    // Now load the model in the popup context
    console.log('Loading model in popup context...');
    if (isWhisperModel) {
      await loadWhisperModel();
    } else {
      await loadStandardModel();
    }
    
    // Verify model loaded correctly
    if (!transcriber || typeof transcriber !== 'function') {
      throw new Error('Transcriber not properly initialized');
    }
    
    console.log('Model loaded successfully, transcriber type:', typeof transcriber);
    statusEl.textContent = 'Model loaded';
    if (modelInfoEl) {
      modelInfoEl.textContent = `Model: ${selectedModel}`;
    }
    progressBar.style.width = '100%';
    
    // Fade out progress bar
    setTimeout(() => {
      progressBar.style.width = '0%';
    }, 1000);
    
  } catch (error) {
    console.error('Model loading error:', error);
    console.error('Error details:', error.message);
    console.error('Error stack:', error.stack);
    if (statusEl) {
      statusEl.textContent = 'Error loading model';
      statusEl.classList.add('error');
    }
    throw error; // Re-throw to be caught by the initialization
  }
}

// Load Whisper model using the specific approach
async function loadWhisperModel() {
  console.log('Loading Whisper model using specific approach');
  
  try {
    // Import the specific model classes
    const { AutoProcessor, AutoModelForSpeechSeq2Seq, WhisperTokenizer } = await import('@xenova/transformers');
    
    // Load the processor
    console.log('Loading processor...');
    processor = await AutoProcessor.from_pretrained(selectedModel, {
      progress_callback: progressCallback,
      cache_dir: env.cacheDir,
      local_files_only: false,
      chunk_length_s: 30,      // Set chunking globally
      stride_length_s: 10,     // Set stride globally
      return_timestamps: true  // Enable timestamps
    });
    
    // Also load the tokenizer explicitly
    console.log('Loading tokenizer...');
    const tokenizer = await WhisperTokenizer.from_pretrained(selectedModel, {
      progress_callback: progressCallback,
      cache_dir: env.cacheDir,
      local_files_only: false,
    });
    
    // Attach the tokenizer to the processor for easier access
    processor.tokenizer = tokenizer;
    console.log('Tokenizer loaded and attached to processor');
    
    // Load the model
    console.log('Loading model...');
    model = await AutoModelForSpeechSeq2Seq.from_pretrained(selectedModel, {
      progress_callback: progressCallback,
      cache_dir: env.cacheDir,
      local_files_only: false,
      chunk_length_s: 30,      // Set chunking globally
      stride_length_s: 10,     // Set stride globally
      return_timestamps: true  // Enable timestamps
    });
    
    // Attach the tokenizer to the model as well
    model.tokenizer = tokenizer;
    
    // Log the processor and model structure
    console.log('Processor structure:', Object.keys(processor));
    console.log('Model structure:', Object.keys(model));
    
    // Add a helper function to properly process audio
    async function processAudioBlob(audioBlob) {
      console.log('Processing audio blob:', audioBlob.size, 'bytes', audioBlob.type);
      
      // Create an audio context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000 // Whisper models expect 16kHz audio
      });
      
      // Convert blob to array buffer
      const arrayBuffer = await audioBlob.arrayBuffer();
      console.log('Audio array buffer size:', arrayBuffer.byteLength);
      
      // Decode the audio data
      try {
        console.log('Decoding audio data...');
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        console.log('Audio decoded successfully:', 
          'duration:', audioBuffer.duration, 
          'channels:', audioBuffer.numberOfChannels,
          'sample rate:', audioBuffer.sampleRate);
        
        // Get the audio data from the first channel
        const audioData = audioBuffer.getChannelData(0);
        console.log('Audio data extracted, length:', audioData.length);
        
        // Resample to 16kHz if needed
        let finalAudioData = audioData;
        if (audioBuffer.sampleRate !== 16000) {
          console.log('Resampling audio from', audioBuffer.sampleRate, 'Hz to 16000 Hz');
          // Simple resampling by taking every nth sample
          const ratio = audioBuffer.sampleRate / 16000;
          const newLength = Math.floor(audioData.length / ratio);
          finalAudioData = new Float32Array(newLength);
          for (let i = 0; i < newLength; i++) {
            finalAudioData[i] = audioData[Math.floor(i * ratio)];
          }
          console.log('Resampled audio data length:', finalAudioData.length);
        }
        
        return finalAudioData;
      } catch (error) {
        console.error('Error decoding audio:', error);
        throw error;
      } finally {
        // Close the audio context
        if (audioContext.state !== 'closed') {
          await audioContext.close();
        }
      }
    }
    
    // Create a custom transcriber function
    transcriber = async (audioBlob, options = {}) => {
      try {
        console.log('Whisper transcriber called with audio blob:', audioBlob.size, 'bytes');
        
        // Process the audio using our helper function
        const audioData = await processAudioBlob(audioBlob);
        
        // Check if processor is valid
        if (!processor) {
          throw new Error('Processor is not initialized');
        }
        
        // Log processor methods to debug
        console.log('Processor methods:', Object.getOwnPropertyNames(processor));
        console.log('Processor prototype methods:', processor.constructor ? Object.getOwnPropertyNames(processor.constructor.prototype) : 'No constructor');
        
        // Estimate audio duration (rough estimate based on sample rate and length)
        const estimatedDuration = audioData.length / 16000; // assuming 16kHz sample rate
        console.log(`Estimated audio duration from samples: ${estimatedDuration.toFixed(2)} seconds`);
        
        // For longer audio, use our own chunking approach
        if (estimatedDuration > 25) {
          console.log('Audio is longer than 25 seconds, using custom chunking approach');
          
          // Merge options with defaults
          const chunkOptions = {
            chunk_length_s: options.chunk_length_s || 20,     // Reduced from 30 to 20 for faster processing
            stride_length_s: options.stride_length_s || 5,    // Reduced from 10 to 5 for better accuracy
            max_new_tokens: options.max_new_tokens || 448,
            return_timestamps: options.return_timestamps !== undefined ? options.return_timestamps : true
          };
          
          console.log('Using custom chunking with options:', chunkOptions);
          
          // Calculate chunk sizes in samples
          const sampleRate = 16000;
          const chunkSizeSamples = chunkOptions.chunk_length_s * sampleRate;
          const strideSizeSamples = chunkOptions.stride_length_s * sampleRate;
          
          // Optimize number of chunks - use larger chunks for very long audio
          let adjustedChunkSize = chunkSizeSamples;
          let adjustedStrideSize = strideSizeSamples;
          
          // For very long audio (>2 minutes), use larger chunks with less overlap
          if (estimatedDuration > 120) {
            adjustedChunkSize = 25 * sampleRate; // 25 seconds
            adjustedStrideSize = 3 * sampleRate; // 3 seconds overlap
            console.log('Very long audio detected, using larger chunks with less overlap');
          }
          
          const numChunks = Math.ceil((audioData.length - adjustedChunkSize) / (adjustedChunkSize - adjustedStrideSize)) + 1;
          
          console.log(`Splitting audio into ${numChunks} chunks of ${adjustedChunkSize/sampleRate}s with ${adjustedStrideSize/sampleRate}s stride`);
          
          // Show progress in the UI
          statusEl.textContent = `Transcribing long audio (${estimatedDuration.toFixed(0)}s)...`;
          
          // Process each chunk and collect results
          let fullTranscription = '';
          let lastWords = new Set(); // Track recently seen words to avoid repetition
          
          for (let i = 0; i < numChunks; i++) {
            const startSample = i * (adjustedChunkSize - adjustedStrideSize);
            const endSample = Math.min(startSample + adjustedChunkSize, audioData.length);
            
            console.log(`Processing chunk ${i+1}/${numChunks}: samples ${startSample} to ${endSample}`);
            
            // Update UI with progress
            statusEl.textContent = `Transcribing chunk ${i+1}/${numChunks}...`;
            
            // Extract chunk data
            const chunkData = audioData.slice(startSample, endSample);
            
            // Process this chunk
            const processorOptions = {
              chunk_length_s: chunkOptions.chunk_length_s,
              stride_length_s: chunkOptions.stride_length_s
            };
            
            const inputs = await processor(chunkData, processorOptions);
            
            // Generate transcription for this chunk
            const generateOptions = {
              max_new_tokens: chunkOptions.max_new_tokens,
              return_timestamps: chunkOptions.return_timestamps
            };
            
            const output = await model.generate(inputs.input_features, generateOptions);
            
            // Decode the output
            let chunkTranscription;
            if (processor.tokenizer && typeof processor.tokenizer.decode === 'function') {
              chunkTranscription = processor.tokenizer.decode(output[0]);
            } else {
              chunkTranscription = model.tokenizer.decode(output[0]);
            }
            
            // Clean up the transcription
            const cleanedChunk = cleanWhisperOutput(chunkTranscription);
            console.log(`Chunk ${i+1} transcription:`, cleanedChunk);
            
            // Add to full transcription with deduplication
            if (cleanedChunk) {
              // Split into words and filter out duplicates from recent chunks
              const words = cleanedChunk.split(/\s+/);
              const uniqueWords = [];
              
              for (const word of words) {
                // Skip if this is a repeated word we've seen recently
                // But allow up to 3 repetitions (for cases like "49, 49, 49" that might be legitimate)
                const repetitionCount = [...lastWords].filter(w => w === word).length;
                if (repetitionCount < 3) {
                  uniqueWords.push(word);
                  // Add to our tracking set
                  lastWords.add(word);
                  // Keep the set at a reasonable size
                  if (lastWords.size > 20) {
                    lastWords.delete([...lastWords][0]);
                  }
                }
              }
              
              // Only add non-empty chunks
              if (uniqueWords.length > 0) {
                const dedupedChunk = uniqueWords.join(' ');
                fullTranscription += (fullTranscription && !fullTranscription.endsWith(' ') ? ' ' : '') + dedupedChunk;
                
                // Update the transcription area with progress
                transcriptionEl.value = fullTranscription + ' [transcribing...]';
              }
            }
          }
          
          console.log('Full transcription from chunks:', fullTranscription);
          return { text: fullTranscription || '[Empty transcription]' };
        }
        
        // For shorter audio, use the standard approach
        console.log('Processing audio with processor and chunking parameters...');
        
        // Always set chunking parameters for consistency
        const processorOptions = {
          chunk_length_s: 30,   // Increased from 15 to 30
          stride_length_s: 10   // Increased from 5 to 10
        };
        
        console.log('Using chunking parameters:', processorOptions);
        
        // Process the audio with options
        const inputs = await processor(audioData, processorOptions);
        console.log('Audio processed, inputs:', inputs);
        
        // Check if model is valid
        if (!model) {
          throw new Error('Model is not initialized');
        }
        
        // Generate the transcription with chunking parameters
        console.log('Generating transcription with model...');
        
        // Always use chunking parameters for consistency
        const generateOptions = {
          max_new_tokens: 448,
          chunk_length_s: 30,   // Increased from 15 to 30
          stride_length_s: 10,  // Increased from 5 to 10
          return_timestamps: true // Add timestamps to help with chunking
        };
        
        console.log('Using generation options:', generateOptions);
        
        const output = await model.generate(inputs.input_features, generateOptions);
        console.log('Transcription generated, output:', output);
        
        // Check if output is valid
        if (!output || !output[0]) {
          throw new Error('Model output is invalid');
        }
        
        // Decode the output - use tokenizer.decode instead of processor.decode if available
        console.log('Decoding output...');
        let transcription;
        
        // Try different decoding methods
        if (typeof processor.decode === 'function') {
          console.log('Using processor.decode method');
          transcription = processor.decode(output[0]);
        } else if (processor.tokenizer && typeof processor.tokenizer.decode === 'function') {
          console.log('Using processor.tokenizer.decode method');
          transcription = processor.tokenizer.decode(output[0]);
        } else if (model.tokenizer && typeof model.tokenizer.decode === 'function') {
          console.log('Using model.tokenizer.decode method');
          transcription = model.tokenizer.decode(output[0]);
        } else {
          // Fallback: try to import the tokenizer directly
          console.log('Using fallback decoding method');
          const { WhisperTokenizer } = await import('@xenova/transformers');
          const tokenizer = await WhisperTokenizer.from_pretrained(selectedModel);
          transcription = tokenizer.decode(output[0]);
        }
        
        console.log('Transcription decoded:', transcription);
        
        // Clean up the transcription by removing special tags
        const cleanedTranscription = cleanWhisperOutput(transcription);
        console.log('Cleaned transcription:', cleanedTranscription);
        
        return { text: cleanedTranscription || '[Empty transcription]' };
      } catch (error) {
        console.error('Error during Whisper transcription:', error);
        console.error('Error details:', error.message);
        console.error('Error stack:', error.stack);
        throw error;
      }
    };
    
    console.log('Whisper model loaded successfully');
  } catch (error) {
    console.error('Error loading Whisper model:', error);
    throw error;
  }
}

// Load standard model using the pipeline approach
async function loadStandardModel() {
  console.log('Loading standard model using pipeline approach');
  
  try {
    transcriber = await pipeline(
      'automatic-speech-recognition', 
      selectedModel, 
      { 
        progress_callback: progressCallback,
        quantized: false,
        revision: 'main',
        framework: 'onnx',
        cache_dir: env.cacheDir,
        chunk_length_s: 30,     // Increased from 15 to 30
        stride_length_s: 10,    // Increased from 5 to 10
        max_new_tokens: 448,    // Increase token limit for longer transcriptions
        return_timestamps: true // Enable timestamps to help with chunking
      }
    );
    
    console.log('Standard model loaded successfully');
  } catch (error) {
    console.error('Error loading standard model:', error);
    throw error;
  }
}

// Format timer
function formatTime(ms) {
  const seconds = Math.floor((ms / 1000) % 60).toString().padStart(2, '0');
  const minutes = Math.floor((ms / 1000 / 60) % 60).toString().padStart(2, '0');
  return `${minutes}:${seconds}`;
}

// Update recording timer
function updateTimer() {
  if (recording) {
    const elapsed = Date.now() - recordingStartTime;
    timerEl.textContent = formatTime(elapsed);
  }
}

// Initialize visualizer
function initVisualizer() {
  if (visualizerInitialized) return;
  
  canvas = document.getElementById('waveform');
  canvasCtx = canvas.getContext('2d');
  
  // Set canvas size
  function resizeCanvas() {
    canvas.width = canvas.offsetWidth * window.devicePixelRatio;
    canvas.height = canvas.offsetHeight * window.devicePixelRatio;
    canvasCtx.scale(window.devicePixelRatio, window.devicePixelRatio);
  }
  
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 2048;
  dataArray = new Uint8Array(analyser.frequencyBinCount);
  
  visualizerInitialized = true;
}

// Draw waveform
function drawWaveform() {
  if (!recording) {
    if (animationId) {
      cancelAnimationFrame(animationId);
      animationId = null;
    }
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  
  animationId = requestAnimationFrame(drawWaveform);
  analyser.getByteTimeDomainData(dataArray);
  
  canvasCtx.fillStyle = 'rgba(0, 0, 0, 0.05)';
  canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
  
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = '#4a6ee0';
  canvasCtx.beginPath();
  
  const sliceWidth = canvas.width / dataArray.length;
  let x = 0;
  
  for (let i = 0; i < dataArray.length; i++) {
    const v = dataArray[i] / 128.0;
    const y = (v * canvas.height) / 2;
    
    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }
    
    x += sliceWidth;
  }
  
  canvasCtx.lineTo(canvas.width, canvas.height / 2);
  canvasCtx.stroke();
}

// Toggle recording
async function toggleRecording() {
  if (!recording) {
    try {
      if (!transcriber) {
        statusEl.textContent = 'Model not loaded';
        return;
      }
      
      audioChunks = [];
      
      try {
        console.log('Requesting microphone access...');
        const stream = await navigator.mediaDevices.getUserMedia({ 
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true,
            sampleRate: 16000
          },
          video: false
        });
        console.log('Microphone access granted');
        
        // Initialize visualizer if needed
        if (!visualizerInitialized) {
          initVisualizer();
        }
        
        // Connect audio stream to visualizer
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        
        // Start waveform animation
        drawWaveform();
        
        // Get available mime types
        const mimeTypes = MediaRecorder.isTypeSupported ? 
          ['audio/webm;codecs=opus', 'audio/webm', 'audio/ogg;codecs=opus', 'audio/wav', 'audio/mp4'].filter(type => MediaRecorder.isTypeSupported(type)) : 
          ['audio/webm'];
        
        console.log('Supported audio MIME types:', mimeTypes);
        
        // Use the first supported mime type
        const options = {
          mimeType: mimeTypes[0],
          audioBitsPerSecond: 128000
        };
        
        console.log('Creating MediaRecorder with options:', options);
        mediaRecorder = new MediaRecorder(stream, options);
        console.log('MediaRecorder created with MIME type:', mediaRecorder.mimeType);
        
        // Set up a timer to request data at regular intervals (every 2 seconds)
        // This helps prevent losing data if the recording is long
        const CHUNK_INTERVAL = 2000; // 2 seconds in milliseconds (reduced from 3s)
        let chunkTimer = null;
        
        // Set a maximum recording time (10 minutes)
        const MAX_RECORDING_TIME = 10 * 60 * 1000; // 10 minutes in milliseconds
        let maxRecordingTimer = null;
        
        mediaRecorder.addEventListener('dataavailable', (event) => {
          if (event.data.size > 0) {
            audioChunks.push(event.data);
            console.log(`Audio chunk received: ${event.data.size} bytes, total chunks: ${audioChunks.length}, total size: ${audioChunks.reduce((sum, chunk) => sum + chunk.size, 0)} bytes`);
          } else {
            console.warn('Received empty audio chunk');
          }
        });
        
        mediaRecorder.addEventListener('start', () => {
          console.log('MediaRecorder started at:', new Date().toISOString());
          recordBtn.innerHTML = '<i class="fas fa-stop"></i><span>Stop Recording</span>';
          
          // Request data every 2 seconds to avoid losing data
          chunkTimer = setInterval(() => {
            if (mediaRecorder && mediaRecorder.state === 'recording') {
              console.log('Requesting data chunk at:', new Date().toISOString());
              mediaRecorder.requestData();
            }
          }, CHUNK_INTERVAL);
          
          // Set a maximum recording time limit
          maxRecordingTimer = setTimeout(() => {
            if (recording && mediaRecorder && mediaRecorder.state === 'recording') {
              console.log('Maximum recording time reached (10 minutes), stopping recording');
              statusEl.textContent = 'Maximum recording time reached (10 minutes)';
              
              // Stop the recording
              stopRecording();
            }
          }, MAX_RECORDING_TIME);
        });
        
        mediaRecorder.addEventListener('stop', async () => {
          console.log('MediaRecorder stopped at:', new Date().toISOString());
          recordBtn.innerHTML = '<i class="fas fa-microphone"></i><span>Start Recording</span>';
          statusEl.classList.add('transcribing');
          statusEl.textContent = 'Transcribing...';
          
          // Clear the timers
          if (chunkTimer) {
            clearInterval(chunkTimer);
            chunkTimer = null;
          }
          
          if (maxRecordingTimer) {
            clearTimeout(maxRecordingTimer);
            maxRecordingTimer = null;
          }
          
          // Make sure we update the recording state and UI
          recording = false;
          recordBtn.classList.remove('recording');
          if (recordingTimer) {
            clearInterval(recordingTimer);
            recordingTimer = null;
          }
          timerEl.textContent = '00:00';
          
          statusEl.textContent = 'Transcribing...';
          
          try {
            // Check if we have any audio data
            if (audioChunks.length === 0) {
              throw new Error('No audio data captured');
            }
            
            const audioBlob = new Blob(audioChunks);
            console.log('Audio blob created:', audioBlob.size, 'bytes', audioBlob.type);
            console.log('Total recording duration:', ((Date.now() - recordingStartTime) / 1000).toFixed(1), 'seconds');
            console.log('Total audio chunks:', audioChunks.length);
            
            // Always process longer recordings in chunks for better results
            const estimatedDuration = (Date.now() - recordingStartTime) / 1000;
            console.log(`Estimated recording duration: ${estimatedDuration.toFixed(1)} seconds`);
            
            // Lower the threshold for chunking to ensure we handle the 30-second limit
            if (estimatedDuration > 10 && audioBlob.size > 50000) {
              console.log('Long recording detected, processing in chunks...');
              await processLongRecording(audioBlob);
            } else {
              // Process normally for shorter recordings
              await processRecording(audioBlob);
            }
            
            statusEl.textContent = 'Transcription complete';
            statusEl.classList.add('success');
            setTimeout(() => {
              statusEl.classList.remove('success');
            }, 2000);
          } catch (error) {
            statusEl.textContent = 'Error transcribing audio';
            statusEl.classList.add('error');
            console.error('Transcription error:', error);
            console.error('Error details:', error.message);
            console.error('Error stack:', error.stack);
            
            // Add the error to the transcription area for visibility
            transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + 
              `[Transcription error: ${error.message}]`;
          }
        });
        
        // Start recording with a timeslice to ensure we get data regularly
        // This is in addition to our manual requestData calls
        mediaRecorder.start(3000); // Request data every 3 seconds automatically (reduced from 5s)
        
        recording = true;
        recordBtn.classList.add('recording');
        statusEl.textContent = 'Recording...';
        recordingStartTime = Date.now();
        recordingTimer = setInterval(updateTimer, 1000);
      } catch (micError) {
        console.error('Microphone access error details:', micError);
        statusEl.textContent = 'Microphone access denied';
        statusEl.classList.add('error');
        
        // Show more helpful message to the user with specific Chrome instructions
        if (micError.name === 'NotAllowedError') {
          alert('Microphone access was denied.\n\nTo enable microphone access:\n1. Click the lock/site settings icon in the address bar\n2. Find "Microphone" and change it to "Allow"\n3. Close and reopen the extension popup\n\nIf that doesn\'t work, you may need to:\n1. Go to chrome://extensions\n2. Find this extension and click "Details"\n3. Ensure "Site access" includes "On all sites"');
        } else if (micError.name === 'NotFoundError') {
          alert('No microphone found. Please connect a microphone and try again.');
        } else {
          alert(`Microphone error: ${micError.message}`);
        }
      }
    } catch (error) {
      statusEl.textContent = 'Error accessing microphone';
      statusEl.classList.add('error');
      console.error('General error:', error);
    }
  } else {
    // Stop recording
    stopRecording();
  }
}

// Helper function to safely stop recording
function stopRecording() {
  console.log('Stopping recording...');
  
  // Safety check to prevent multiple stop calls
  if (!recording) {
    console.log('Recording already stopped');
    return;
  }
  
  try {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      console.log('Stopping MediaRecorder');
      mediaRecorder.stop();
      
      // Stop all tracks in the stream
      if (mediaRecorder.stream) {
        console.log('Stopping all tracks in the stream');
        mediaRecorder.stream.getTracks().forEach(track => {
          console.log(`Stopping track: ${track.kind}`);
          track.stop();
        });
      }
    } else {
      console.log('MediaRecorder already inactive or not initialized');
      
      // Force UI update in case the stop event doesn't fire
      recording = false;
      recordBtn.innerHTML = '<i class="fas fa-microphone"></i><span>Start Recording</span>';
      recordBtn.classList.remove('recording');
      if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
      }
      timerEl.textContent = '00:00';
      statusEl.textContent = 'Ready';
    }
  } catch (error) {
    console.error('Error stopping recording:', error);
    
    // Force UI update even if there's an error
    recording = false;
    recordBtn.innerHTML = '<i class="fas fa-microphone"></i><span>Start Recording</span>';
    recordBtn.classList.remove('recording');
    if (recordingTimer) {
      clearInterval(recordingTimer);
      recordingTimer = null;
    }
    timerEl.textContent = '00:00';
    statusEl.textContent = 'Error stopping recording';
    statusEl.classList.add('error');
  }
}

// Process a normal (shorter) recording
async function processRecording(audioBlob) {
  try {
    // Log audio details
    console.log('Processing recording directly, blob size:', audioBlob.size, 'bytes');
    const audioArrayBuffer = await audioBlob.arrayBuffer();
    console.log('Audio array buffer size:', audioArrayBuffer.byteLength);
    
    // Check if we have a valid transcriber function
    console.log('Transcriber type:', typeof transcriber);
    if (typeof transcriber !== 'function') {
      throw new Error('Transcriber is not a function. Model may not be properly loaded.');
    }
    
    // Check if this is actually a long recording that should be chunked
    const estimatedDuration = (Date.now() - recordingStartTime) / 1000;
    console.log(`Estimated recording duration: ${estimatedDuration.toFixed(1)} seconds`);
    
    // If it's longer than 25 seconds, use the long recording processor instead
    if (estimatedDuration > 25) {
      console.log('Recording is longer than 25 seconds, redirecting to long recording processor');
      return await processLongRecording(audioBlob);
    }
    
    // Try the main transcription method first
    console.log('Starting transcription with main method...');
    
    // For short recordings, we'll try to process them directly first
    const result = await transcriber(audioBlob);
    statusEl.classList.remove('transcribing');
    
    if (!result || !result.text) {
      console.warn('Transcription result is empty or invalid:', result);
      transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + '[No speech detected]';
    } else {
      // Clean the text before adding it
      const cleanedText = cleanWhisperOutput(result.text);
      
      // Only add non-empty text
      if (cleanedText.trim()) {
        transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + cleanedText;
        console.log('Added transcription:', cleanedText);
      } else {
        console.warn('Cleaned text was empty');
        transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + '[No speech detected]';
      }
    }
  } catch (error) {
    statusEl.classList.remove('transcribing');
    // If the main method fails, try the fallback method
    console.error('Main transcription method failed:', error);
    console.log('Trying fallback transcription method...');
    
    try {
      // Import the pipeline directly
      const { pipeline } = await import('@xenova/transformers');
      
      // Create a temporary pipeline with specific chunk settings
      const tempTranscriber = await pipeline(
        'automatic-speech-recognition',
        selectedModel,
        {
          quantized: false,
          revision: 'main',
          framework: 'onnx',
          chunk_length_s: Math.min(estimatedDuration, 20), // Use smaller chunks if needed
          stride_length_s: 1 // Small overlap between chunks
        }
      );
      
      // Try transcribing with the temporary pipeline
      const fallbackResult = await tempTranscriber(audioBlob);
      console.log('Fallback transcription result:', fallbackResult);
      
      if (fallbackResult && fallbackResult.text) {
        // Clean up the fallback result as well
        const cleanedFallbackText = cleanWhisperOutput(fallbackResult.text);
        
        // Only add non-empty text
        if (cleanedFallbackText.trim()) {
          transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + cleanedFallbackText;
          console.log('Added fallback transcription:', cleanedFallbackText);
        } else {
          console.warn('Cleaned fallback text was empty');
          transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + '[No speech detected]';
        }
      } else {
        throw new Error('Fallback transcription failed');
      }
    } catch (fallbackError) {
      console.error('Fallback transcription also failed:', fallbackError);
      
      // Try one more approach - process in smaller chunks
      try {
        console.log('Trying to process as a long recording as last resort...');
        await processLongRecording(audioBlob);
      } catch (lastError) {
        console.error('All transcription methods failed:', lastError);
        throw error; // Throw the original error
      }
    }
  }
}

// Process a long recording by splitting it into chunks
async function processLongRecording(audioBlob) {
  try {
    console.log(`Starting long recording process for blob of size: ${audioBlob.size} bytes`);
    console.log(`Recording duration was approximately: ${((Date.now() - recordingStartTime) / 1000).toFixed(1)} seconds`);
    
    // Validate the audio blob before processing
    if (!audioBlob || audioBlob.size === 0) {
      console.error('Invalid audio blob: empty or zero size');
      throw new Error('Empty audio recording');
    }
    
    // Check if we have a valid transcriber
    if (!transcriber || typeof transcriber !== 'function') {
      console.error('Transcriber is not available or not a function');
      throw new Error('Transcriber is not properly initialized');
    }
    
    // Get the estimated duration
    const estimatedDuration = (Date.now() - recordingStartTime) / 1000;
    console.log(`Estimated recording duration: ${estimatedDuration.toFixed(1)} seconds`);
    
    // For very short recordings, just use the direct approach
    if (estimatedDuration < 15) {
      console.log('Recording is short enough to process directly');
      
      try {
        // Try direct transcription first
        console.log('Attempting direct transcription...');
        const result = await transcriber(audioBlob);
        
        if (result && result.text) {
          const cleanedText = cleanWhisperOutput(result.text);
          if (cleanedText) {
            transcriptionEl.value = cleanedText;
            console.log('Direct transcription successful:', cleanedText);
            return;
          }
        }
      } catch (directError) {
        console.warn('Direct transcription failed:', directError);
        // Continue with chunking approach
      }
    }
    
    // For longer recordings, we'll use a manual chunking approach
    console.log('Using manual chunking approach for long recording');
    
    // Create smaller audio blobs from the original
    // We'll use time-based chunking by creating new audio elements
    const CHUNK_SIZE = 10; // seconds
    const numChunks = Math.ceil(estimatedDuration / CHUNK_SIZE);
    console.log(`Splitting audio into ${numChunks} chunks of ${CHUNK_SIZE} seconds each`);
    
    // Create an audio element to play the recording
    const audioElement = new Audio();
    const audioUrl = URL.createObjectURL(audioBlob);
    audioElement.src = audioUrl;
    
    // Wait for metadata to load
    await new Promise(resolve => {
      audioElement.addEventListener('loadedmetadata', resolve);
      audioElement.load();
    });
    
    const audioDuration = audioElement.duration;
    console.log(`Actual audio duration from metadata: ${audioDuration.toFixed(2)} seconds`);
    
    // Process the entire recording with the existing transcriber
    console.log('Processing entire recording with existing transcriber...');
    try {
      // Increase chunking parameters for longer audio
      const transcriptionOptions = {
        chunk_length_s: 30,       // Increased from 15 to 30
        stride_length_s: 10,      // Increased from 5 to 10
        max_new_tokens: 448,
        return_timestamps: true   // Enable timestamps to help with chunking
      };
      
      console.log('Using transcription options:', transcriptionOptions);
      
      // Process with options
      const result = await transcriber(audioBlob, transcriptionOptions);
      console.log('Full transcription result:', result);
      
      if (result && result.text) {
        const cleanedText = cleanWhisperOutput(result.text);
        console.log('Cleaned full transcription:', cleanedText);
        
        if (cleanedText) {
          transcriptionEl.value = cleanedText;
          console.log('Full transcription successful');
          URL.revokeObjectURL(audioUrl);
          return;
        }
      }
    } catch (fullError) {
      console.warn('Full transcription failed:', fullError);
      // Continue with fallback approach
    }
    
    // If we get here, we need to try a different approach
    // Let's try using the Web Audio API to manually split the audio
    console.log('Trying Web Audio API approach...');
    
    try {
      // Create an audio context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Fetch the audio data
      const response = await fetch(audioUrl);
      const arrayBuffer = await response.arrayBuffer();
      
      // Decode the audio
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      console.log('Audio decoded successfully:', 
        'duration:', audioBuffer.duration.toFixed(2), 'seconds',
        'channels:', audioBuffer.numberOfChannels,
        'sample rate:', audioBuffer.sampleRate);
      
      // Extract audio data for processing
      // Convert to mono and get Float32Array
      const audioData = new Float32Array(audioBuffer.length);
      const leftChannel = audioBuffer.getChannelData(0);
      
      // If stereo, average with right channel, otherwise just use left
      if (audioBuffer.numberOfChannels > 1) {
        const rightChannel = audioBuffer.getChannelData(1);
        for (let i = 0; i < audioBuffer.length; i++) {
          audioData[i] = (leftChannel[i] + rightChannel[i]) / 2;
        }
      } else {
        // Just copy mono channel
        audioData.set(leftChannel);
      }
      
      // Process in chunks directly using audio data
      let fullTranscription = '';
      // Use larger chunks for manual processing
      const CHUNK_SIZE = 30; // seconds
      const sampleRate = audioBuffer.sampleRate;
      const samplesPerChunk = CHUNK_SIZE * sampleRate;
      const totalSamples = audioData.length;
      const numChunks = Math.ceil(totalSamples / samplesPerChunk);
      
      console.log(`Processing audio in ${numChunks} chunks of ${CHUNK_SIZE} seconds each`);
      
      // Process each chunk
      for (let i = 0; i < numChunks; i++) {
        const startTime = i * samplesPerChunk;
        const endTime = Math.min((i + 1) * samplesPerChunk, totalSamples);
        const chunkLength = endTime - startTime;
        
        console.log(`Processing chunk ${i+1}/${numChunks}: ${startTime.toFixed(1)}s to ${endTime.toFixed(1)}s (${chunkLength.toFixed(1)}s)`);
        
        // Skip very short chunks
        if (chunkLength < 0.5) {
          console.log(`Skipping chunk ${i+1} as it's too short`);
          continue;
        }
        
        // Create a new buffer for this chunk
        const chunkBuffer = audioContext.createBuffer(
          audioBuffer.numberOfChannels,
          Math.floor(chunkLength * audioBuffer.sampleRate),
          audioBuffer.sampleRate
        );
        
        // Copy the audio data for this chunk
        for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
          const channelData = audioBuffer.getChannelData(channel);
          const chunkData = chunkBuffer.getChannelData(channel);
          
          const startSample = Math.floor(startTime * audioBuffer.sampleRate);
          const endSample = Math.floor(endTime * audioBuffer.sampleRate);
          
          for (let j = 0; j < (endSample - startSample); j++) {
            chunkData[j] = channelData[startSample + j];
          }
        }
        
        // Convert the chunk to a blob
        const chunkBlob = await audioBufferToBlob(chunkBuffer);
        console.log(`Chunk ${i+1} converted to blob: ${chunkBlob.size} bytes`);
        
        // Update status
        statusEl.textContent = `Transcribing part ${i+1}/${numChunks}...`;
        
        try {
          // Transcribe this chunk using our existing transcriber
          const result = await transcriber(chunkBlob);
          console.log(`Transcription result for chunk ${i+1}:`, result);
          
          if (result && result.text) {
            const chunkText = cleanWhisperOutput(result.text);
            console.log(`Chunk ${i+1} cleaned transcription:`, chunkText);
            
            if (chunkText) {
              fullTranscription += (fullTranscription ? ' ' : '') + chunkText;
              transcriptionEl.value = fullTranscription + ' [transcribing...]';
            }
          }
        } catch (chunkError) {
          console.error(`Error transcribing chunk ${i+1}:`, chunkError);
          // Continue with next chunk
        }
      }
      
      // Clean up
      audioContext.close();
      URL.revokeObjectURL(audioUrl);
      
      // Update final transcription
      if (fullTranscription) {
        transcriptionEl.value = fullTranscription;
        console.log('Chunked transcription complete');
      } else {
        transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + '[No speech detected]';
        console.warn('No speech detected in any chunks');
      }
    } catch (audioError) {
      console.error('Error processing audio with Web Audio API:', audioError);
      URL.revokeObjectURL(audioUrl);
      
      // Final fallback: just show the error
      transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + 
        `[Error processing audio: ${audioError.message}]`;
    }
  } catch (error) {
    console.error('Error in processLongRecording:', error);
    console.error('Error details:', error.message);
    console.error('Error stack:', error.stack);
    
    // Add error to transcription
    transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + 
      `[Error processing recording: ${error.message}]`;
    
    throw error;
  }
}

// Helper function to convert AudioBuffer to Blob
async function audioBufferToBlob(audioBuffer) {
  // Create an offline audio context
  const offlineContext = new OfflineAudioContext(
    audioBuffer.numberOfChannels,
    audioBuffer.length,
    audioBuffer.sampleRate
  );
  
  // Create a buffer source
  const source = offlineContext.createBufferSource();
  source.buffer = audioBuffer;
  source.connect(offlineContext.destination);
  source.start(0);
  
  // Render the audio
  const renderedBuffer = await offlineContext.startRendering();
  
  // Convert to WAV
  return bufferToWav(renderedBuffer);
}

// Convert an AudioBuffer to a WAV blob
function bufferToWav(buffer) {
  const numChannels = buffer.numberOfChannels;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const bytesPerSample = bitDepth / 8;
  const blockAlign = numChannels * bytesPerSample;
  
  // Create the WAV file
  const dataLength = buffer.length * numChannels * bytesPerSample;
  const fileLength = 44 + dataLength;
  
  const arrayBuffer = new ArrayBuffer(fileLength);
  const view = new DataView(arrayBuffer);
  
  // RIFF chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, fileLength - 8, true);
  writeString(view, 8, 'WAVE');
  
  // FMT sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // subchunk size
  view.setUint16(20, format, true); // audio format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true); // byte rate
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, bitDepth, true);
  
  // Data sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataLength, true);
  
  // Write the PCM samples
  const data = new Float32Array(buffer.getChannelData(0));
  let offset = 44;
  
  for (let i = 0; i < data.length; i++) {
    const sample = Math.max(-1, Math.min(1, data[i]));
    const value = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
    view.setInt16(offset, value, true);
    offset += 2;
  }
  
  return new Blob([view], { type: 'audio/wav' });
}

// Helper to write a string to a DataView
function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

// Copy transcription to clipboard
function copyTranscription() {
  if (!transcriptionEl.value) return;
  transcriptionEl.select();
  document.execCommand('copy');
  // Alternative modern approach:
  // navigator.clipboard.writeText(transcriptionEl.value);
  statusEl.textContent = 'Copied to clipboard';
  statusEl.classList.add('success');
  setTimeout(() => {
    statusEl.textContent = 'Ready';
    statusEl.classList.remove('success');
  }, 2000);
}

// Clear transcription
function clearTranscription() {
  transcriptionEl.value = '';
  statusEl.textContent = 'Cleared';
  setTimeout(() => {
    statusEl.textContent = 'Ready';
  }, 1000);
}

// Change model
async function changeModel() {
  selectedModel = modelSelect.value;
  // Clear old model
  transcriber = null;
  processor = null;
  model = null;
  await initializeModel();
}

// Save settings
function saveSettings() {
  chrome.storage.local.set({
    selectedModel: selectedModel
  });
}

// Load settings
async function loadSettings() {
  const settings = await chrome.storage.local.get(['selectedModel']);
  if (settings.selectedModel) {
    selectedModel = settings.selectedModel;
    modelSelect.value = selectedModel;
  }
}

// Event listeners
// Remove old event listeners first to prevent duplicates
recordBtn.removeEventListener('click', toggleRecording);
copyBtn.removeEventListener('click', copyTranscription);
clearBtn.removeEventListener('click', clearTranscription);
modelSelect.removeEventListener('change', changeModel);

// Add event listeners
recordBtn.addEventListener('click', async (e) => {
  console.log('Record button clicked, current state:', {
    recording,
    transcriber: !!transcriber,
    model: !!model,
    processor: !!processor
  });
  await toggleRecording();
});
copyBtn.addEventListener('click', copyTranscription);
clearBtn.addEventListener('click', clearTranscription);
modelSelect.addEventListener('change', changeModel);

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  console.log('Popup initialized, starting setup...');
  try {
    // Disable record button and show loading state
    recordBtn.disabled = true;
    recordBtn.classList.add('loading');
    statusEl.textContent = 'Loading model...';

    // Load settings first
    console.log('Loading settings...');
    await loadSettings();
    
    // Initialize model
    console.log('Settings loaded, initializing model...');
    await initializeModel();
    
    // Enable button and update UI
    console.log('Model initialized successfully');
    recordBtn.disabled = false;
    recordBtn.classList.remove('loading');
    statusEl.textContent = 'Ready';

    // Re-bind event listeners
    console.log('Binding event listeners...');
    recordBtn.addEventListener('click', async () => {
      console.log('Record button clicked, current state:', {
        recording,
        transcriber: !!transcriber,
        model: !!model,
        processor: !!processor
      });
      await toggleRecording();
    });
    copyBtn.addEventListener('click', copyTranscription);
    clearBtn.addEventListener('click', clearTranscription);
    modelSelect.addEventListener('change', changeModel);
  } catch (error) {
    console.error('Initialization error:', error);
    statusEl.textContent = 'Error loading model';
    statusEl.classList.add('error');
    recordBtn.disabled = true;
    recordBtn.classList.remove('loading');
  }
});

// Save settings when popup closes
window.addEventListener('unload', saveSettings); 