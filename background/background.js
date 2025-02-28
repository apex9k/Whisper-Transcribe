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

// Listen for extension icon clicks to open side panel
chrome.action.onClicked.addListener((tab) => {
  // Open side panel when extension icon is clicked
  chrome.sidePanel.open({ tabId: tab.id });
});

// Listen for installation
chrome.runtime.onInstalled.addListener((details) => {
  console.log('Extension installed:', details);
  
  if (details.reason === 'install') {
    // Set initial settings on install
    chrome.storage.local.set({
      defaultModel: 'Xenova/whisper-tiny.en',
      preloadModel: true,
      darkMode: false,
      lastActiveTab: 'transcribe',
      transcriptionHistory: []
    });
  }
});

// Listen for messages from popup/side panel
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('Background script received message:', message);
  
  if (message.action === 'ping') {
    sendResponse({ status: 'active' });
    return true;
  }
  
  if (message.action === 'preloadModel') {
    console.log('Preloading model:', message.model);
    // In a full implementation, you would preload the model here
    sendResponse({ status: 'success', message: 'Model preloaded (simulated)' });
    return true;
  }
  
  if (message.action === 'saveTranscription') {
    saveToHistory(message.text, message.model);
    sendResponse({ status: 'success' });
    return true;
  }
});

// Save a transcription to history
async function saveToHistory(text, model) {
  if (!text || text.trim() === '') return;
  
  try {
    // Get current history
    const data = await chrome.storage.local.get('transcriptionHistory');
    const history = data.transcriptionHistory || [];
    
    // Add new entry
    history.unshift({
      id: Date.now(),
      text: text,
      model: model,
      timestamp: new Date().toISOString(),
      preview: text.substring(0, 100) + (text.length > 100 ? '...' : '')
    });
    
    // Limit history to 50 entries
    if (history.length > 50) {
      history.pop();
    }
    
    // Save back to storage
    await chrome.storage.local.set({ transcriptionHistory: history });
    console.log('Saved to history, total entries:', history.length);
  } catch (error) {
    console.error('Error saving to history:', error);
  }
}

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