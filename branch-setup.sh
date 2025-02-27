#!/bin/bash

# Function to display script usage
show_usage() {
    echo "Usage: ./branch-setup.sh [dev|prod]"
    echo "  dev  - Switch to development branch with unpacked permissions"
    echo "  prod - Switch to production branch with Chrome Web Store permissions"
}

# Check if argument is provided
if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

# Function to update manifest for development
update_manifest_dev() {
    cat > manifest.json << EOL
{
  "manifest_version": 3,
  "name": "Whisper Transcribe (Dev)",
  "version": "1.0.0",
  "description": "Real-time speech transcription using Whisper AI - 100% private, runs locally in your browser",
  "permissions": [
    "storage",
    "unlimitedStorage",
    "activeTab"
  ],
  "host_permissions": [
    "https://huggingface.co/*",
    "https://*.huggingface.co/*"
  ],
  "content_security_policy": {
    "extension_pages": "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'"
  },
  "action": {
    "default_popup": "popup/popup.html",
    "default_icon": {
      "16": "assets/icon16.png",
      "48": "assets/icon48.png",
      "128": "assets/icon128.png"
    }
  },
  "icons": {
    "16": "assets/icon16.png",
    "48": "assets/icon48.png",
    "128": "assets/icon128.png"
  },
  "background": {
    "service_worker": "background/service-worker.js",
    "type": "module"
  },
  "web_accessible_resources": [{
    "resources": ["models/*"],
    "matches": ["<all_urls>"]
  }]
}
EOL
}

# Function to update manifest for production
update_manifest_prod() {
    cat > manifest.json << EOL
{
  "manifest_version": 3,
  "name": "Whisper Transcribe",
  "version": "1.0.0",
  "description": "Real-time speech transcription using Whisper AI - 100% private, runs locally in your browser",
  "permissions": [
    "storage",
    "activeTab"
  ],
  "host_permissions": [
    "*://*/*"
  ],
  "action": {
    "default_popup": "popup/popup.html",
    "default_icon": {
      "16": "assets/icon16.png",
      "48": "assets/icon48.png",
      "128": "assets/icon128.png"
    }
  },
  "icons": {
    "16": "assets/icon16.png",
    "48": "assets/icon48.png",
    "128": "assets/icon128.png"
  },
  "background": {
    "service_worker": "background/service-worker.js",
    "type": "module"
  },
  "web_accessible_resources": [{
    "resources": ["models/*"],
    "matches": ["<all_urls>"]
  }]
}
EOL
}

# Handle branch switching
case "$1" in
    "dev")
        git checkout -b dev 2>/dev/null || git checkout dev
        update_manifest_dev
        echo "Switched to development branch with unpacked permissions"
        ;;
    "prod")
        git checkout -b main 2>/dev/null || git checkout main
        update_manifest_prod
        echo "Switched to production branch with Chrome Web Store permissions"
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

# Stage manifest changes
git add manifest.json
git status 