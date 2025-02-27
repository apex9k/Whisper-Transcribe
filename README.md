# Whisper Transcribe Chrome Extension

A Chrome extension for real-time speech transcription using Whisper AI, powered by [🤗 transformers.js](https://github.com/huggingface/transformers.js). This extension performs transcription entirely locally in your browser - your audio never leaves your device!

![Whisper Transcribe Demo](assets/demo.gif)
[📦 Install from Chrome Web Store](#) *(Coming Soon)*

## Features

- 🎙️ Real-time speech transcription
- 💻 100% local processing - your audio never leaves your device
- 🔒 Privacy-focused design
- 🌐 Multiple language support
- ⚡ Fast and efficient using WebGPU acceleration
- 🎯 Easy-to-use interface
- 📋 Copy transcriptions to clipboard
- 🎚️ Multiple Whisper model options:
  - Tiny (English)
  - Small (English)
  - Base (English)
  - Tiny (Multilingual)
  - Small (Multilingual)

## Demo Video
*(Coming Soon)*

## Installation

1. [Install from Chrome Web Store](#) *(Coming Soon)*
2. Click the extension icon in your browser
3. Allow microphone access when prompted
4. Start transcribing!

## Development

To run this extension locally:

1. Clone this repository
2. Open Chrome and navigate to `chrome://extensions/`
3. Enable "Developer mode"
4. Click "Load unpacked" and select the extension directory

## Technical Details

This extension is built using:
- [transformers.js](https://github.com/huggingface/transformers.js) - For running Whisper AI models in the browser
- Web Audio API - For audio processing
- Chrome Extension Manifest V3

The extension is inspired by the [WebGPU Whisper example](https://github.com/huggingface/transformers.js/tree/main/examples/webgpu-whisper) from the transformers.js repository.

## Privacy

This extension processes all audio locally in your browser. No audio data or transcriptions are ever sent to external servers. The only network requests made are to download the Whisper model files when you first select them.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

This project is derived from and inspired by the [transformers.js](https://github.com/huggingface/transformers.js) project, which is also licensed under the Apache License 2.0.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [🤗 Hugging Face](https://huggingface.co/) for transformers.js
- [OpenAI](https://openai.com/) for the Whisper model
- The transformers.js team for their excellent WebGPU example 