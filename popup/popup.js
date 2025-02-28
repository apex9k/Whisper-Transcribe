import { pipeline, env } from '@xenova/transformers';
import { loadWhisperModel, loadStandardModel, cleanWhisperOutput } from '../shared/whisper-utils.js';

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
    // Reset UI
    progressBar.style.width = '0%';
    
    // If we're using the same model as before and it's already loaded, skip
    if (transcriber && selectedModel === modelSelect.value) {
      console.log('Model already loaded:', selectedModel);
      statusEl.textContent = 'Ready';
      return { success: true };
    }
    
    // Update model selection and store preference
    selectedModel = modelSelect.value;
    chrome.storage.local.set({ defaultModel: selectedModel });
    
    // Disable record button during loading
    if (recordBtn) recordBtn.disabled = true;
    
    // Check if backend is ready
    try {
      const response = await chrome.runtime.sendMessage({ action: 'ping' });
      console.log('Background service worker status:', response);
      
      // Try to use preloaded model from background
      const preloadResponse = await chrome.runtime.sendMessage({
        action: 'preloadModel',
        model: selectedModel
      });
      
      console.log('Model preloaded by background script:', preloadResponse);
    } catch (error) {
      console.warn('Background service worker not ready:', error);
    }
    
    // Update UI
    statusEl.textContent = 'Loading model...';
    
    // Check if it's a Whisper model (naming pattern)
    isWhisperModel = selectedModel.toLowerCase().includes('whisper');
    
    // Load appropriate model type using shared utility functions
    let result;
    if (isWhisperModel) {
      console.log('Loading Whisper model');
      result = await loadWhisperModel(selectedModel, progressCallback);
      processor = result.processor;
      model = result.model;
      transcriber = result.transcriber;
    } else {
      console.log('Loading standard model');
      result = await loadStandardModel(selectedModel, progressCallback);
      transcriber = result.transcriber;
    }
    
    console.log('Model loaded successfully');
    
    // Update UI
    statusEl.textContent = 'Ready';
    if (modelInfoEl) {
      modelInfoEl.textContent = `Model: ${selectedModel}`;
    }
    progressBar.style.width = '100%';
    
    // Fade out progress bar after a delay
    setTimeout(() => {
      progressBar.style.width = '0%';
    }, 1000);
    
    // Re-enable record button
    if (recordBtn) recordBtn.disabled = false;
    
    return { success: true };
  } catch (error) {
    console.error('Error initializing model:', error);
    statusEl.textContent = 'Error loading model: ' + error.message;
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
  
  // Save the transcription to history
  saveTranscriptionToHistory();
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

// Initialize UI
document.addEventListener('DOMContentLoaded', function() {
  // Initialize tabs
  const tabButtons = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  
  tabButtons.forEach(button => {
    button.addEventListener('click', () => {
      const tabName = button.getAttribute('data-tab');
      
      // Update active tab button
      tabButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
      
      // Show corresponding tab pane
      tabPanes.forEach(pane => {
        pane.classList.remove('active');
        if (pane.id === `${tabName}-tab`) {
          pane.classList.add('active');
        }
      });
      
      // Save preference
      chrome.storage.local.set({ lastActiveTab: tabName });
    });
  });
  
  // Initialize size toggles
  const sizeButtons = document.querySelectorAll('.size-btn');
  
  sizeButtons.forEach(button => {
    button.addEventListener('click', () => {
      const size = button.getAttribute('data-size');
      
      // Update active size button
      sizeButtons.forEach(btn => btn.classList.remove('active'));
      button.classList.add('active');
      
      // Apply size to body
      document.body.classList.remove('size-small', 'size-medium', 'size-large');
      document.body.classList.add(`size-${size}`);
      
      // Save preference
      chrome.storage.local.set({ preferredSize: size });
    });
  });
  
  // Initialize side panel button
  const sidePanelBtn = document.getElementById('sidepanel-btn');
  if (sidePanelBtn) {
    sidePanelBtn.addEventListener('click', () => {
      chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
        if (tabs && tabs[0]) {
          chrome.sidePanel.open({ tabId: tabs[0].id });
          window.close(); // Close the popup
        }
      });
    });
  }
  
  // Initialize accordions in help tab
  const accordionHeaders = document.querySelectorAll('.accordion-header');
  
  accordionHeaders.forEach(header => {
    header.addEventListener('click', () => {
      const content = header.nextElementSibling;
      
      // Toggle active state
      header.classList.toggle('active');
      content.classList.toggle('active');
    });
  });
  
  // Initialize auto-expanding textarea
  const transcriptionArea = document.getElementById('transcription');
  if (transcriptionArea) {
    transcriptionArea.addEventListener('input', function() {
      // Reset height to auto to get the correct scrollHeight
      this.style.height = 'auto';
      // Set height to scrollHeight to expand the textarea
      this.style.height = (this.scrollHeight) + 'px';
    });
  }
  
  // Load user preferences
  loadUserPreferences();
});

// Load user preferences from storage
async function loadUserPreferences() {
  try {
    const prefs = await chrome.storage.local.get(['preferredSize', 'lastActiveTab', 'darkMode']);
    
    // Apply size preference
    if (prefs.preferredSize) {
      const sizeButtons = document.querySelectorAll('.size-btn');
      sizeButtons.forEach(btn => {
        if (btn.getAttribute('data-size') === prefs.preferredSize) {
          btn.click(); // Simulate click to apply the size
        }
      });
    }
    
    // Apply tab preference
    if (prefs.lastActiveTab) {
      const tabButtons = document.querySelectorAll('.tab-btn');
      tabButtons.forEach(btn => {
        if (btn.getAttribute('data-tab') === prefs.lastActiveTab) {
          btn.click(); // Simulate click to switch to the tab
        }
      });
    }
    
    // Apply dark mode preference
    if (prefs.darkMode) {
      document.body.classList.add('dark-mode');
      const darkModeCheckbox = document.getElementById('dark-mode');
      if (darkModeCheckbox) {
        darkModeCheckbox.checked = true;
      }
    }
  } catch (error) {
    console.error('Error loading preferences:', error);
  }
}

// Handle dark mode toggle
const darkModeCheckbox = document.getElementById('dark-mode');
if (darkModeCheckbox) {
  darkModeCheckbox.addEventListener('change', function() {
    if (this.checked) {
      document.body.classList.add('dark-mode');
      chrome.storage.local.set({ darkMode: true });
    } else {
      document.body.classList.remove('dark-mode');
      chrome.storage.local.set({ darkMode: false });
    }
  });
}

// Handle settings save
const saveSettingsBtn = document.getElementById('save-settings');
if (saveSettingsBtn) {
  saveSettingsBtn.addEventListener('click', async function() {
    const defaultModel = document.getElementById('default-model').value;
    const preloadModel = document.getElementById('preload-model').checked;
    
    // Save settings
    await chrome.storage.local.set({
      defaultModel,
      preloadModel
    });
    
    // Update model if changed
    if (defaultModel !== selectedModel) {
      selectedModel = defaultModel;
      modelSelect.value = defaultModel;
      await changeModel();
    }
    
    // Show success message
    statusEl.textContent = 'Settings saved';
    statusEl.classList.add('success');
    
    // Switch back to transcribe tab
    setTimeout(() => {
      const transcribeTab = document.querySelector('[data-tab="transcribe"]');
      if (transcribeTab) {
        transcribeTab.click();
      }
    }, 1500);
  });
}

// Add a function to save transcription to history
async function saveTranscriptionToHistory() {
  const text = transcriptionEl.value.trim();
  if (!text) return;
  
  try {
    await chrome.runtime.sendMessage({
      action: 'saveTranscription',
      text: text,
      model: selectedModel
    });
    console.log('Transcription saved to history');
  } catch (error) {
    console.error('Error saving transcription:', error);
  }
} 