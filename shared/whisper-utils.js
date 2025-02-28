import { pipeline, env } from '@xenova/transformers';

// Configure transformers.js to use the extension's storage
env.localModelPath = chrome.runtime.getURL('models/');
env.allowRemoteModels = true;
env.useBrowserCache = true;
env.cacheDir = 'transformers-cache';

// Configure ONNX runtime settings
env.backends.onnx.wasm.numThreads = 1;
env.backends.onnx.wasm.simd = false;

/**
 * Loads a Whisper model using specialized approach
 * @param {string} modelName - Name of the model to load
 * @param {Function} progressCallback - Callback for progress updates
 * @returns {Object} - Object containing processor and model
 */
export async function loadWhisperModel(modelName, progressCallback) {
  try {
    // Import model-specific classes
    const { AutoProcessor, AutoModelForSpeechSeq2Seq, AutoTokenizer, read_audio } = await import('@xenova/transformers');
    
    // Store read_audio function for later use
    window.read_audio = read_audio;
    
    // Load the processor and tokenizer
    console.log('Loading processor and tokenizer...');
    const [processor, tokenizer] = await Promise.all([
      AutoProcessor.from_pretrained(modelName, {
        progress_callback: progressCallback,
        cache_dir: env.cacheDir,
        local_files_only: false,
      }),
      AutoTokenizer.from_pretrained(modelName, {
        progress_callback: progressCallback,
        cache_dir: env.cacheDir,
        local_files_only: false,
      })
    ]);
    
    // Load the model
    console.log('Loading model...');
    const model = await AutoModelForSpeechSeq2Seq.from_pretrained(modelName, {
      progress_callback: progressCallback,
      cache_dir: env.cacheDir,
      local_files_only: false,
    });
    
    // Create a transcription function
    const transcriber = async function(audioBlob) {
      try {
        console.log('Processing audio blob:', audioBlob.size, 'bytes', audioBlob.type);
        
        // Convert blob to array buffer
        const arrayBuffer = await audioBlob.arrayBuffer();
        console.log('Audio array buffer size:', arrayBuffer.byteLength);
        
        // Process audio data properly depending on the type
        let audioData;
        
        try {
          // Try using read_audio directly with the array buffer (this is safer than using fetch)
          audioData = await read_audio(arrayBuffer, 16000);
          console.log('Successfully processed audio data using read_audio directly');
        } catch (audioReadError) {
          console.error('Error using read_audio directly:', audioReadError);
          
          // Fallback: process audio manually
          try {
            console.log('Falling back to manual audio processing');
            
            // Create an audio context
            const audioContext = new (window.AudioContext || window.webkitAudioContext)({
              sampleRate: 16000 // Whisper models expect 16kHz audio
            });
            
            // Decode the audio data
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            console.log('Audio decoded successfully:', 
              'duration:', audioBuffer.duration, 
              'channels:', audioBuffer.numberOfChannels,
              'sample rate:', audioBuffer.sampleRate);
            
            // Get the audio data from the first channel
            const channelData = audioBuffer.getChannelData(0);
            
            // Resample to 16kHz if needed
            if (audioBuffer.sampleRate !== 16000) {
              console.log('Resampling audio from', audioBuffer.sampleRate, 'Hz to 16000 Hz');
              const ratio = audioBuffer.sampleRate / 16000;
              const newLength = Math.floor(channelData.length / ratio);
              audioData = new Float32Array(newLength);
              for (let i = 0; i < newLength; i++) {
                audioData[i] = channelData[Math.floor(i * ratio)];
              }
            } else {
              audioData = channelData;
            }
            
            await audioContext.close();
            console.log('Manual audio processing complete');
          } catch (manualError) {
            console.error('Manual audio processing failed:', manualError);
            throw manualError;
          }
        }
        
        console.log('Final audio data length:', audioData.length);
        
        // Process audio with model
        console.log('Processing audio with model...');
        
        // Check that processor is available
        if (!processor) {
          console.error('Processor is unavailable:', processor);
          throw new Error('Whisper processor not available for transcription');
        }
        
        console.log('Processor type:', typeof processor);
        
        // The processor might be a callable function wrapper
        // It might look like: ƒ (...t){return c._call(...t)}
        try {
          // Pass audioData directly as Float32Array
          console.log('Calling processor...');
          const inputs = await processor(audioData);
          console.log('Processor call succeeded, output structure:', Object.keys(inputs));
          
          // Generate using the input_features from the processor output
          console.log('Calling model.generate with inputs.input_features');
          const output = await model.generate(inputs.input_features, {
            max_new_tokens: 128,
            return_timestamps: false,
          });
          
          console.log('Model output type:', typeof output, Array.isArray(output) ? 'Array' : 'Not an array');
          if (output && output[0]) {
            console.log('Output[0] type:', typeof output[0]);
            if (typeof output[0] === 'object') {
              console.log('Output[0] properties:', Object.keys(output[0]));
              console.log('Output[0] methods:', Object.getOwnPropertyNames(Object.getPrototypeOf(output[0])));
            }
          }
          
          // Find a suitable decoder
          let decoder = tokenizer; // Use the tokenizer we loaded earlier
          let canDecode = typeof decoder.decode === 'function';
          
          if (!canDecode) {
            console.log('Primary tokenizer not available, trying alternatives...');
            
            // Try model.tokenizer
            if (model.tokenizer && typeof model.tokenizer.decode === 'function') {
              console.log('Using model.tokenizer for decoding');
              decoder = model.tokenizer;
              canDecode = true;
            }
            // Try processor.tokenizer
            else if (processor.tokenizer && typeof processor.tokenizer.decode === 'function') {
              console.log('Using processor.tokenizer for decoding');
              decoder = processor.tokenizer;
              canDecode = true;
            }
            // Try processor directly
            else if (typeof processor.decode === 'function') {
              console.log('Using processor.decode directly');
              decoder = processor;
              canDecode = true;
            }
          } else {
            console.log('Using primary tokenizer for decoding');
          }
          
          if (!canDecode) {
            console.error('No decoder available - this will result in raw output only');
          }
          
          // Decode the output - handle different output formats
          let result;
          try {
            console.log('Attempting to decode output', 'decode method available:', canDecode);
            
            // Get the data from the output tensor
            let outputData;
            
            if (output[0]) {
              // Try different ways to get the data
              if (output[0].tolist && typeof output[0].tolist === 'function') {
                outputData = output[0].tolist();
              } else if (output[0].array && typeof output[0].array === 'function') {
                outputData = output[0].array();
              } else if (output[0].data && Array.isArray(output[0].data)) {
                outputData = output[0].data;  
              } else if (Array.isArray(output[0])) {
                outputData = output[0];
              } else {
                // If all else fails, try to use the output directly
                outputData = output[0];
              }
            } else {
              outputData = output;
            }
            
            console.log('Output data type for decoding:', typeof outputData, 
                      Array.isArray(outputData) ? 'Array' : 'Not an array');
            
            // Safely decode with error handling
            if (canDecode) {
              result = decoder.decode(outputData, { skip_special_tokens: true });
            } else {
              console.error('No decode function available after all attempts');
              
              // Emergency fallback: Convert output to JSON string as a last resort
              result = `[Unable to decode properly] Raw output: ${JSON.stringify(outputData).substring(0, 100)}...`;
            }
          } catch (decodeError) {
            console.error('Error decoding output:', decodeError);
            
            // Last resort, return a simple representation of the output
            if (Array.isArray(output) && output.length > 0) {
              result = `[Decode failed] Raw output available but couldn't be processed`;
            } else {
              result = '[No transcription available due to decoding error]';
            }
          }
          
          console.log('Transcription result:', result);
          return { text: result || '[Empty transcription result]' };
          
        } catch (processingError) {
          console.error('Error processing audio with model:', processingError);
          return { text: `[Processing error: ${processingError.message}]`, error: processingError };
        }
      } catch (error) {
        console.error('Error in transcription function:', error);
        return { text: `[Error during transcription: ${error.message}]`, error };
      }
    };
    
    console.log('Whisper model loaded successfully');
    return { processor, model, transcriber, tokenizer };
  } catch (error) {
    console.error('Error loading Whisper model:', error);
    throw error;
  }
}

/**
 * Loads a standard ASR model using pipeline
 * @param {string} modelName - Name of the model to load
 * @param {Function} progressCallback - Callback for progress updates
 * @returns {Function} - Transcription function
 */
export async function loadStandardModel(modelName, progressCallback) {
  try {
    // Use the pipeline function to load the model
    const transcriber = await pipeline(
      'automatic-speech-recognition',
      modelName,
      {
        quantized: false,
        revision: 'main',
        framework: 'onnx',
        progress_callback: progressCallback
      }
    );
    
    console.log('Standard model loaded successfully');
    return { transcriber };
  } catch (error) {
    console.error('Error loading standard model:', error);
    throw error;
  }
}

/**
 * Clean Whisper output by removing special tags
 * @param {string} text - Raw transcription text
 * @returns {string} - Cleaned text
 */
export function cleanWhisperOutput(text) {
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
    .replace(/\[BLANK_AUDIO\]/g, '')
    .replace(/^\s*\[\s*music\s*\]\s*/i, '')
    .replace(/\[\s*music\s*\]\s*$/i, '')
    .replace(/\s+/g, ' ') // Normalize whitespace
    .replace(/^\s*\.\s*$/, '') // Remove lone periods
    .replace(/^[.,;:!?]+/, '') // Remove leading punctuation
    .replace(/[.,;:!?]+$/, '') // Remove trailing punctuation
    .trim();
} 