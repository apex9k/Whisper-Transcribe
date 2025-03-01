// Import from our shared utility module
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
let currentHistoryItem = null;

// Add visualizer variables
let audioContext;
let analyser;
let dataArray;
let canvas;
let canvasCtx;
let animationId;
let visualizerInitialized = false;

// DOM elements - we'll populate on DOM load
let recordBtn, transcriptionEl, copyBtn, clearBtn, modelSelect, statusEl, timerEl, modelInfoEl, progressBar;
let historyListEl, historyDetailEl, historyCopyBtn, historyRestoreBtn, historyDeleteBtn, clearHistoryBtn;

// Initialize DOM elements once document is loaded
document.addEventListener('DOMContentLoaded', function() {
  // Get DOM elements for transcribe tab
  recordBtn = document.getElementById('record-btn');
  transcriptionEl = document.getElementById('transcription');
  copyBtn = document.getElementById('copy-btn');
  clearBtn = document.getElementById('clear-btn');
  modelSelect = document.getElementById('model-select');
  statusEl = document.getElementById('status');
  timerEl = document.getElementById('timer');
  modelInfoEl = document.getElementById('model-info');
  progressBar = document.getElementById('progress-bar');
  
  // Get DOM elements for history tab
  historyListEl = document.getElementById('history-list');
  historyDetailEl = document.getElementById('history-detail');
  historyCopyBtn = document.getElementById('history-copy-btn');
  historyRestoreBtn = document.getElementById('history-restore-btn');
  historyDeleteBtn = document.getElementById('history-delete-btn');
  clearHistoryBtn = document.getElementById('clear-history-btn');
  
  // Set up event listeners for transcribe tab
  recordBtn.addEventListener('click', toggleRecording);
  copyBtn.addEventListener('click', copyTranscription);
  clearBtn.addEventListener('click', clearTranscription);
  modelSelect.addEventListener('change', changeModel);
  
  // Set up event listeners for history tab
  if (historyCopyBtn) historyCopyBtn.addEventListener('click', copyHistoryItem);
  if (historyRestoreBtn) historyRestoreBtn.addEventListener('click', restoreHistoryItem);
  if (historyDeleteBtn) historyDeleteBtn.addEventListener('click', deleteHistoryItem);
  if (clearHistoryBtn) clearHistoryBtn.addEventListener('click', clearHistory);
  
  // Initialize the UI
  initializeUI();
  
  // Initialize the model
  initializeModel().catch(error => {
    console.error('Initialization error:', error);
    statusEl.textContent = 'Error: ' + error.message;
    statusEl.classList.add('error');
    recordBtn.disabled = true;
  });
  
  // Load transcription history
  loadTranscriptionHistory();
});

// Initialize UI for the side panel
function initializeUI() {
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
  
  // Initialize theme toggle
  const themeToggle = document.getElementById('theme-toggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', () => {
      document.body.classList.toggle('dark-mode');
      const isDarkMode = document.body.classList.contains('dark-mode');
      chrome.storage.local.set({ darkMode: isDarkMode });
      
      // Update icon
      themeToggle.innerHTML = isDarkMode 
        ? '<i class="fas fa-sun"></i>' 
        : '<i class="fas fa-moon"></i>';
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
  if (transcriptionEl) {
    transcriptionEl.addEventListener('input', function() {
      // Reset height to auto to get the correct scrollHeight
      this.style.height = 'auto';
      // Set height to scrollHeight to expand the textarea
      this.style.height = (this.scrollHeight) + 'px';
    });
  }
  
  // Add save button functionality
  const saveBtn = document.getElementById('save-btn');
  if (saveBtn) {
    saveBtn.addEventListener('click', () => {
      const text = transcriptionEl.value;
      if (!text) return;
      
      // Save to history
      saveTranscriptionToHistory();
      
      // Create a blob with the text content
      const blob = new Blob([text], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      
      // Create a download link
      const a = document.createElement('a');
      a.href = url;
      a.download = `transcription-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.txt`;
      a.click();
      
      // Clean up
      URL.revokeObjectURL(url);
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
  
  // Load user preferences
  loadUserPreferences();
}

// Load user preferences from storage
async function loadUserPreferences() {
  try {
    const prefs = await chrome.storage.local.get(['lastActiveTab', 'darkMode', 'defaultModel']);
    
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
      const themeToggle = document.getElementById('theme-toggle');
      if (themeToggle) {
        themeToggle.innerHTML = '<i class="fas fa-sun"></i>';
      }
    }
    
    // Apply default model preference
    if (prefs.defaultModel) {
      selectedModel = prefs.defaultModel;
      if (modelSelect) {
        modelSelect.value = prefs.defaultModel;
      }
    }
  } catch (error) {
    console.error('Error loading preferences:', error);
  }
}

// Progress callback for model loading
const progressCallback = (progress) => {
  if (progress.status === 'progress') {
    loadingProgress = Math.round(progress.progress * 100);
    statusEl.textContent = `Loading model: ${loadingProgress}%`;
    progressBar.style.width = `${loadingProgress}%`;
  } else if (progress.status === 'init') {
    statusEl.textContent = 'Initializing model...';
  } else if (progress.status === 'download') {
    loadingProgress = Math.round((progress.loaded / progress.total) * 100);
    statusEl.textContent = `Downloading model: ${loadingProgress}%`;
    progressBar.style.width = `${loadingProgress}%`;
  }
};

// Initialize model based on model type
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

// Implement full toggle recording function
async function toggleRecording() {
  if (!recording) {
    try {
      // Start recording
      recording = true;
      
      // Update button UI
      recordBtn.classList.add('recording');
      recordBtn.querySelector('span').textContent = 'Stop Recording';
      recordBtn.querySelector('i').className = 'fas fa-stop';
      
      // Clear previous transcription
      if (transcriptionEl.value.trim() === '') {
        transcriptionEl.value = '';
      }
      
      // Reset timer
      recordingStartTime = Date.now();
      if (recordingTimer) clearInterval(recordingTimer);
      recordingTimer = setInterval(updateTimer, 1000);
      timerEl.textContent = '00:00';
      
      // Update status
      statusEl.textContent = 'Recording...';
      statusEl.classList.add('transcribing');
      
      // Initialize audio context and analyzer
      if (!visualizerInitialized) {
        initVisualizer();
      }
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      
      // Connect to visualizer
      if (audioContext && analyser) {
        const source = audioContext.createMediaStreamSource(stream);
        source.connect(analyser);
        drawWaveform();
      }
      
      // Set up media recorder
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];
      
      // Collect audio chunks
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      // Handle recording stop
      mediaRecorder.onstop = async () => {
        // Stop the visualizer
        if (animationId) {
          cancelAnimationFrame(animationId);
          animationId = null;
        }
        
        // Process the recording
        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
        audioChunks = [];
        
        try {
          // Process audio and get transcription
          await processRecording(audioBlob);
        } catch (error) {
          console.error('Error processing recording:', error);
          statusEl.textContent = 'Error: ' + error.message;
          statusEl.classList.add('error');
          statusEl.classList.remove('transcribing');
        }
        
        // Clean up
        if (recordingTimer) {
          clearInterval(recordingTimer);
          recordingTimer = null;
        }
        
        // Reset button state
        recordBtn.classList.remove('recording');
        recordBtn.querySelector('span').textContent = 'Start Recording';
        recordBtn.querySelector('i').className = 'fas fa-microphone';
        recording = false;
        
        // Release all media streams
        stream.getTracks().forEach(track => track.stop());
      };
      
      // Start recording
      mediaRecorder.start(1000); // Collect data every second
    } catch (error) {
      console.error('Error starting recording:', error);
      statusEl.textContent = 'Error: ' + error.message;
      statusEl.classList.add('error');
      statusEl.classList.remove('transcribing');
      
      // Reset recording state
      recording = false;
      recordBtn.classList.remove('recording');
      recordBtn.querySelector('span').textContent = 'Start Recording';
      recordBtn.querySelector('i').className = 'fas fa-microphone';
    }
  } else {
    // Stop recording
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
      mediaRecorder.stop();
      statusEl.textContent = 'Processing...';
      recordBtn.disabled = true;
      recordBtn.classList.add('transcribing');
      recordBtn.querySelector('span').textContent = 'Transcribing...';
    }
  }
}

// Process the recorded audio
async function processRecording(audioBlob) {
  try {
    statusEl.textContent = 'Transcribing...';
    
    // If we have a valid transcriber function
    if (transcriber && typeof transcriber === 'function') {
      // Transcribe the audio
      const result = await transcriber(audioBlob);
      
      if (result && result.text) {
        // Clean the output (remove special tokens)
        const cleanedText = cleanWhisperOutput(result.text);
        
        // Append to textarea or replace
        if (transcriptionEl.value) {
          transcriptionEl.value += (transcriptionEl.value.endsWith('.') ? ' ' : '. ') + cleanedText;
        } else {
          transcriptionEl.value = cleanedText;
        }
        
        // Auto-expand textarea in case of auto-resize
        if (transcriptionEl.dispatchEvent) {
          transcriptionEl.dispatchEvent(new Event('input'));
        }
        
        statusEl.textContent = 'Ready';
        statusEl.classList.remove('transcribing');
        
        // Reset button state
        recordBtn.disabled = false;
        recordBtn.classList.remove('transcribing');
        recordBtn.querySelector('span').textContent = 'Start Recording';
        
        // Also save to history when completing
        saveTranscriptionToHistory();
        
        return cleanedText;
      } else {
        throw new Error('No transcription result');
      }
    } else {
      throw new Error('Transcriber not available');
    }
  } catch (error) {
    console.error('Transcription error:', error);
    statusEl.textContent = 'Error: ' + (error.message || 'Failed to transcribe');
    statusEl.classList.add('error');
    statusEl.classList.remove('transcribing');
  }
}

// Update the timer display
function updateTimer() {
  const elapsedMs = Date.now() - recordingStartTime;
  timerEl.textContent = formatTime(elapsedMs);
}

// Initialize audio visualizer
function initVisualizer() {
  canvas = document.getElementById('waveform');
  if (!canvas) return;
  
  canvasCtx = canvas.getContext('2d');
  
  function resizeCanvas() {
    const visualizerContainer = document.querySelector('.visualizer-container');
    if (visualizerContainer) {
      canvas.width = visualizerContainer.clientWidth;
      canvas.height = visualizerContainer.clientHeight;
    }
  }
  
  // Handle resize
  resizeCanvas();
  window.addEventListener('resize', resizeCanvas);
  
  // Initialize audio context
  audioContext = new (window.AudioContext || window.webkitAudioContext)();
  analyser = audioContext.createAnalyser();
  analyser.fftSize = 256;
  
  dataArray = new Uint8Array(analyser.frequencyBinCount);
  
  visualizerInitialized = true;
}

// Draw waveform visualization
function drawWaveform() {
  if (!canvas || !canvasCtx || !analyser) return;
  
  const width = canvas.width;
  const height = canvas.height;
  
  analyser.getByteTimeDomainData(dataArray);
  
  canvasCtx.fillStyle = 'rgba(248, 249, 250, 0.5)';
  canvasCtx.fillRect(0, 0, width, height);
  
  canvasCtx.lineWidth = 2;
  canvasCtx.strokeStyle = '#4a6ee0';
  
  canvasCtx.beginPath();
  
  const sliceWidth = width / dataArray.length;
  let x = 0;
  
  for (let i = 0; i < dataArray.length; i++) {
    const v = dataArray[i] / 128.0;
    const y = v * height / 2;
    
    if (i === 0) {
      canvasCtx.moveTo(x, y);
    } else {
      canvasCtx.lineTo(x, y);
    }
    
    x += sliceWidth;
  }
  
  canvasCtx.lineTo(width, height / 2);
  canvasCtx.stroke();
  
  animationId = requestAnimationFrame(drawWaveform);
}

function formatTime(ms) {
  const seconds = Math.floor(ms / 1000);
  const minutes = Math.floor(seconds / 60);
  return `${minutes.toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
}

function copyTranscription() {
  if (!transcriptionEl.value) return;
  
  navigator.clipboard.writeText(transcriptionEl.value)
    .then(() => {
      copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
      setTimeout(() => {
        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
      }, 2000);
    })
    .catch(err => {
      console.error('Failed to copy text: ', err);
      alert('Failed to copy text. Please try again.');
    });
}

function clearTranscription() {
  transcriptionEl.value = '';
  transcriptionEl.style.height = 'auto';
}

async function changeModel() {
  selectedModel = modelSelect.value;
  chrome.storage.local.set({ defaultModel: selectedModel });
  
  // Reinitialize with the new model
  await initializeModel();
}

// Save current transcription to history
async function saveTranscriptionToHistory() {
  const text = transcriptionEl.value.trim();
  if (!text) return;
  
  try {
    await chrome.runtime.sendMessage({
      action: 'saveTranscription',
      text: text,
      model: selectedModel
    });
    
    // Refresh history if we're on the history tab
    if (document.querySelector('.tab-btn[data-tab="history"]').classList.contains('active')) {
      loadTranscriptionHistory();
    }
  } catch (error) {
    console.error('Error saving transcription:', error);
  }
}

// Load transcription history from storage
async function loadTranscriptionHistory() {
  try {
    const data = await chrome.storage.local.get('transcriptionHistory');
    const history = data.transcriptionHistory || [];
    
    if (!historyListEl) return;
    
    // Clear existing items
    historyListEl.innerHTML = '';
    
    if (history.length === 0) {
      // Show empty state
      historyListEl.innerHTML = `
        <div class="history-empty">
          <i class="fas fa-history"></i>
          <p>No transcriptions yet</p>
        </div>
      `;
      
      // Disable history action buttons
      if (historyCopyBtn) historyCopyBtn.disabled = true;
      if (historyRestoreBtn) historyRestoreBtn.disabled = true;
      if (historyDeleteBtn) historyDeleteBtn.disabled = true;
      if (clearHistoryBtn) clearHistoryBtn.disabled = true;
      
      return;
    }
    
    // Enable clear history button
    if (clearHistoryBtn) clearHistoryBtn.disabled = false;
    
    // Create history items
    history.forEach(item => {
      const historyItem = document.createElement('div');
      historyItem.className = 'history-item';
      historyItem.dataset.id = item.id;
      
      const date = new Date(item.timestamp);
      const formattedDate = date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
      
      // Extract model name from full path
      const modelName = item.model.split('/').pop();
      
      historyItem.innerHTML = `
        <div class="history-item-header">
          <span class="history-item-model">${modelName}</span>
          <span class="history-item-time">${formattedDate}</span>
        </div>
        <div class="history-item-preview">${item.preview}</div>
      `;
      
      historyItem.addEventListener('click', () => {
        // Mark as active
        document.querySelectorAll('.history-item').forEach(el => el.classList.remove('active'));
        historyItem.classList.add('active');
        
        // Show in detail view
        historyDetailEl.value = item.text;
        
        // Enable action buttons
        if (historyCopyBtn) historyCopyBtn.disabled = false;
        if (historyRestoreBtn) historyRestoreBtn.disabled = false;
        if (historyDeleteBtn) historyDeleteBtn.disabled = false;
        
        // Store current item
        currentHistoryItem = item;
      });
      
      historyListEl.appendChild(historyItem);
    });
    
    // Initially disable action buttons until an item is selected
    if (historyCopyBtn) historyCopyBtn.disabled = true;
    if (historyRestoreBtn) historyRestoreBtn.disabled = true;
    if (historyDeleteBtn) historyDeleteBtn.disabled = true;
    
  } catch (error) {
    console.error('Error loading transcription history:', error);
  }
}

// Copy history item to clipboard
function copyHistoryItem() {
  if (!historyDetailEl.value) return;
  
  navigator.clipboard.writeText(historyDetailEl.value)
    .then(() => {
      historyCopyBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
      setTimeout(() => {
        historyCopyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
      }, 2000);
    })
    .catch(err => {
      console.error('Failed to copy text: ', err);
      alert('Failed to copy text. Please try again.');
    });
}

// Restore history item to transcription tab
function restoreHistoryItem() {
  if (!currentHistoryItem) return;
  
  // Set transcription text
  transcriptionEl.value = currentHistoryItem.text;
  
  // Switch to transcribe tab
  document.querySelector('.tab-btn[data-tab="transcribe"]').click();
  
  // Show success message
  statusEl.textContent = 'Transcription restored';
  statusEl.classList.add('success');
  
  // Clear after 2 seconds
  setTimeout(() => {
    statusEl.textContent = 'Ready';
    statusEl.classList.remove('success');
  }, 2000);
}

// Delete history item
async function deleteHistoryItem() {
  if (!currentHistoryItem) return;
  
  try {
    // Get current history
    const data = await chrome.storage.local.get('transcriptionHistory');
    let history = data.transcriptionHistory || [];
    
    // Filter out the current item
    history = history.filter(item => item.id !== currentHistoryItem.id);
    
    // Save back to storage
    await chrome.storage.local.set({ transcriptionHistory: history });
    
    // Clear detail view
    historyDetailEl.value = '';
    
    // Refresh history list
    loadTranscriptionHistory();
    
    // Reset current item
    currentHistoryItem = null;
    
    // Disable action buttons
    if (historyCopyBtn) historyCopyBtn.disabled = true;
    if (historyRestoreBtn) historyRestoreBtn.disabled = true;
    if (historyDeleteBtn) historyDeleteBtn.disabled = true;
    
  } catch (error) {
    console.error('Error deleting history item:', error);
  }
}

// Clear all history
async function clearHistory() {
  if (!confirm('Are you sure you want to clear all transcription history?')) {
    return;
  }
  
  try {
    // Clear history in storage
    await chrome.storage.local.set({ transcriptionHistory: [] });
    
    // Clear detail view
    historyDetailEl.value = '';
    
    // Refresh history list
    loadTranscriptionHistory();
    
    // Reset current item
    currentHistoryItem = null;
    
  } catch (error) {
    console.error('Error clearing history:', error);
  }
} 