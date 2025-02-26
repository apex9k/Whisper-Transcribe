// Background script to handle model pre-loading
// and keep the service worker alive
import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js to use the extension's storage
env.localModelPath = chrome.runtime.getURL('models/');
env.allowRemoteModels = true;  // Allow downloading models from HuggingFace
env.useBrowserCache = true;    // Enable browser cache for model files
env.cacheDir = 'transformers-cache'; // Set a specific cache directory name

// Configure ONNX runtime settings
env.backends.onnx.wasm.numThreads = 1;  // Use single thread for better compatibility
env.backends.onnx.wasm.simd = false;    // Disable SIMD for better compatibility

// This is a minimal background script needed for MV3
console.log('Local Whisper Transcriber background service worker started');
console.log('Model path:', env.localModelPath);
console.log('ONNX configuration:', env.backends.onnx.wasm);

// Create a simple listener to keep the service worker active
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'ping') {
    sendResponse({ status: 'active' });
    return true;
  }
  
  if (message.action === 'preloadModel') {
    // We need to return true immediately to keep the message channel open
    // for the async response
    preloadModel(message.model)
      .then(result => {
        console.log('Preload completed, sending response');
        sendResponse({ status: 'success', result });
      })
      .catch(error => {
        console.error('Preload failed, sending error response');
        sendResponse({ status: 'error', message: error.message });
      });
    return true;
  }
  
  return false;
});

// Preload model function
async function preloadModel(modelName) {
  try {
    console.log('Preloading model:', modelName);
    
    // For Whisper models, we need to use a specific approach
    if (modelName.toLowerCase().includes('whisper')) {
      console.log('Detected Whisper model, using specific loading approach');
      
      // Import the specific model classes directly
      const { AutoProcessor, AutoModelForSpeechSeq2Seq } = await import('@xenova/transformers');
      
      // First, load the processor
      console.log('Loading processor...');
      const processor = await AutoProcessor.from_pretrained(modelName, {
        progress_callback: (progress) => {
          if (progress.status === 'download') {
            console.log(`Downloading processor: ${Math.round(progress.progress * 100)}%`);
          } else if (progress.status === 'init') {
            console.log('Initializing processor...');
          }
        },
        cache_dir: env.cacheDir,
        local_files_only: false,
      });
      console.log('Processor loaded successfully');
      
      // Then, load the model
      console.log('Loading model...');
      const model = await AutoModelForSpeechSeq2Seq.from_pretrained(modelName, {
        progress_callback: (progress) => {
          if (progress.status === 'download') {
            console.log(`Downloading model: ${Math.round(progress.progress * 100)}%`);
          } else if (progress.status === 'init') {
            console.log('Initializing model...');
          }
        },
        cache_dir: env.cacheDir,
        local_files_only: false,
      });
      console.log('Model loaded successfully');
      
      return { success: true, message: 'Model and processor loaded successfully' };
    } else {
      // For non-Whisper models, use the standard approach
      console.log('Using standard pipeline for non-Whisper model');
      const transcriber = await pipeline(
        'automatic-speech-recognition',
        modelName,
        {
          quantized: false,
          revision: 'main',
          framework: 'onnx',
          progress_callback: (progress) => {
            if (progress.status === 'download') {
              console.log(`Downloading model: ${Math.round(progress.progress * 100)}%`);
            } else if (progress.status === 'init') {
              console.log('Initializing model...');
            }
          }
        }
      );
      
      console.log('Model preloaded successfully:', modelName);
      return { success: true, message: 'Pipeline created successfully' };
    }
  } catch (error) {
    console.error('Error preloading model:', error);
    console.error('Error details:', error.message);
    console.error('Error stack:', error.stack);
    throw error;
  }
} 