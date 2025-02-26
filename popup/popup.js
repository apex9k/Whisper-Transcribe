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
    }
    
    // Now load the model in the popup context
    if (isWhisperModel) {
      await loadWhisperModel();
    } else {
      await loadStandardModel();
    }
    
    statusEl.textContent = 'Model loaded';
    modelInfoEl.textContent = `Model: ${selectedModel}`;
    progressBar.style.width = '100%';
    
    // Fade out progress bar
    setTimeout(() => {
      progressBar.style.width = '0%';
    }, 1000);
    
  } catch (error) {
    statusEl.textContent = 'Error loading model';
    statusEl.classList.add('error');
    console.error('Model loading error:', error);
    console.error('Error details:', error.message);
    console.error('Error stack:', error.stack);
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
        .trim();
    }
    
    // Create a custom transcriber function
    transcriber = async (audioBlob) => {
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
        
        // Process the audio
        console.log('Processing audio with processor...');
        const inputs = await processor(audioData);
        console.log('Audio processed, inputs:', inputs);
        
        // Check if model is valid
        if (!model) {
          throw new Error('Model is not initialized');
        }
        
        // Generate the transcription
        console.log('Generating transcription with model...');
        const output = await model.generate(inputs.input_features);
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

// Toggle recording
async function toggleRecording() {
  if (!recording) {
    // Start recording
    try {
      if (!transcriber) {
        statusEl.textContent = 'Model not loaded';
        return;
      }
      
      audioChunks = [];
      
      // Add more detailed error handling for microphone access
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
        
        mediaRecorder.addEventListener('dataavailable', (event) => {
          audioChunks.push(event.data);
        });
        
        mediaRecorder.addEventListener('stop', async () => {
          statusEl.textContent = 'Transcribing...';
          
          try {
            const audioBlob = new Blob(audioChunks);
            console.log('Audio blob created:', audioBlob.size, 'bytes', audioBlob.type);
            
            // Log audio details
            const audioArrayBuffer = await audioBlob.arrayBuffer();
            console.log('Audio array buffer size:', audioArrayBuffer.byteLength);
            
            // Check if we have a valid transcriber function
            console.log('Transcriber type:', typeof transcriber);
            if (typeof transcriber !== 'function') {
              throw new Error('Transcriber is not a function. Model may not be properly loaded.');
            }
            
            // Try the main transcription method first
            try {
              console.log('Starting transcription with main method...');
              const result = await transcriber(audioBlob);
              console.log('Transcription result:', result);
              
              if (!result || !result.text) {
                console.warn('Transcription result is empty or invalid:', result);
                transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + '[No speech detected]';
              } else {
                transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + result.text;
              }
            } catch (transcriptionError) {
              // If the main method fails, try the fallback method
              console.error('Main transcription method failed:', transcriptionError);
              console.log('Trying fallback transcription method...');
              
              try {
                // Import the pipeline directly
                const { pipeline } = await import('@xenova/transformers');
                
                // Create a temporary pipeline
                const tempTranscriber = await pipeline(
                  'automatic-speech-recognition',
                  selectedModel,
                  {
                    quantized: false,
                    revision: 'main',
                    framework: 'onnx'
                  }
                );
                
                // Try transcribing with the temporary pipeline
                const fallbackResult = await tempTranscriber(audioBlob);
                console.log('Fallback transcription result:', fallbackResult);
                
                if (fallbackResult && fallbackResult.text) {
                  // Clean up the fallback result as well
                  const cleanedFallbackText = cleanWhisperOutput(fallbackResult.text);
                  transcriptionEl.value += (transcriptionEl.value ? '\n' : '') + 
                    cleanedFallbackText;
                } else {
                  throw new Error('Fallback transcription failed');
                }
              } catch (fallbackError) {
                console.error('Fallback transcription also failed:', fallbackError);
                throw transcriptionError; // Throw the original error
              }
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
          
          recording = false;
          recordBtn.textContent = 'Start Recording';
          recordBtn.classList.remove('recording');
          clearInterval(recordingTimer);
          timerEl.textContent = '00:00';
        });
        
        mediaRecorder.start();
        recording = true;
        recordBtn.textContent = 'Stop Recording';
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
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
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
recordBtn.addEventListener('click', toggleRecording);
copyBtn.addEventListener('click', copyTranscription);
clearBtn.addEventListener('click', clearTranscription);
modelSelect.addEventListener('change', changeModel);

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
  await loadSettings();
  await initializeModel();
});

// Save settings when popup closes
window.addEventListener('unload', saveSettings); 