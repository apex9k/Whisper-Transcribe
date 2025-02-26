# Local Whisper Transcriber

A Chrome extension that uses Whisper AI to transcribe speech directly in your browser. All processing happens locally - your audio never leaves your device.

## Features

- 100% local speech transcription using Whisper AI models
- Multiple model options (tiny, small, base) in both English and multilingual versions
- Simple recording interface with timer
- Copy and clear transcription functionality
- Persistent model selection

## Technical Details

- Built with vanilla JavaScript
- Uses [@xenova/transformers](https://github.com/xenova/transformers.js) for running Whisper models in the browser
- Implements WebAssembly for efficient model execution
- Handles audio processing and resampling for optimal transcription quality

## Development

### Prerequisites

- Node.js and npm

### Setup

1. Clone the repository
2. Install dependencies:
   ```
   npm install
   ```
3. Build the extension:
   ```
   npm run build
   ```
4. Load the extension in Chrome:
   - Go to `chrome://extensions/`
   - Enable "Developer mode"
   - Click "Load unpacked"
   - Select the `dist` directory

### Project Structure

- `popup/` - Contains the extension popup UI
- `background/` - Contains the service worker for background tasks
- `build.js` - Build script for bundling the extension
- `manifest.json` - Extension manifest file

## Usage

1. Click the extension icon to open the popup
2. Select a model from the dropdown
3. Click "Start Recording" and allow microphone access
4. Speak into your microphone
5. Click "Stop Recording" to end recording and start transcription
6. Use the Copy button to copy the transcription to your clipboard

## License

MIT

## Acknowledgements

- [Transformers.js](https://github.com/xenova/transformers.js)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [Hugging Face](https://huggingface.co/) 