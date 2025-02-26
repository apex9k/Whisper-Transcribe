# Local Whisper Transcriber - Usage Guide

## Loading the Extension in Chrome

1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" by toggling the switch in the top right corner
3. Click "Load unpacked" and select the `dist` folder from this project
4. The extension should now appear in your browser toolbar

## Using the Extension

1. Click on the extension icon in your browser toolbar to open the popup
2. Select a model from the dropdown menu:
   - Tiny models are faster but less accurate
   - Small and Base models are more accurate but slower and use more resources
   - Models with ".en" are optimized for English only
   - Models without ".en" support multiple languages

3. Click "Start Recording" to begin recording your speech
4. Speak clearly into your microphone
5. Click "Stop Recording" when you're finished
6. Wait for the transcription to complete (this may take a few seconds depending on the model size and recording length)
7. The transcribed text will appear in the text area
8. Use the "Copy" button to copy the transcription to your clipboard
9. Use the "Clear" button to clear the transcription if needed

## First-Time Use

When you first use the extension, it will download the selected Whisper model from Hugging Face. This may take some time depending on your internet connection and the size of the model:

- Tiny models: ~75MB
- Small models: ~250MB
- Base models: ~500MB

After the initial download, the model will be cached locally and will load much faster on subsequent uses.

## Troubleshooting

### Microphone Access
- Make sure you've granted microphone access to Chrome
- Check that no other application is using your microphone

### Model Loading Issues
- If the model fails to load, try selecting a smaller model
- Check your internet connection if downloading for the first time
- Try refreshing the extension by clicking the refresh icon on the extensions page

### Transcription Quality
- Speak clearly and at a moderate pace
- Reduce background noise
- Position your microphone closer to the sound source
- Try a larger model for better accuracy

## Privacy

This extension processes all audio locally on your device. The only network requests made are to download the AI model from Hugging Face's servers the first time you use it. After the initial download, the model is cached and no further network requests are made. 