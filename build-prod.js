import { build } from 'esbuild';
import fs from 'fs';
import path from 'path';

// Ensure dist-prod directory exists
if (!fs.existsSync('dist-prod')) {
  fs.mkdirSync('dist-prod');
}

// Ensure subdirectories exist
const subdirs = ['popup', 'background', 'assets', 'models', 'sidepanel', 'shared'];
for (const dir of subdirs) {
  const fullPath = path.join('dist-prod', dir);
  if (!fs.existsSync(fullPath)) {
    fs.mkdirSync(fullPath, { recursive: true });
  }
}

// Create a placeholder file in the models directory
fs.writeFileSync('dist-prod/models/.placeholder', 'This directory is used to store model files.');

// Build the JS files
build({
  entryPoints: [
    'popup/popup.js', 
    'background/background.js',
    'sidepanel/sidepanel.js'
  ],
  bundle: true,
  format: 'esm',
  outdir: 'dist-prod',
  minify: true,
  sourcemap: false, // No sourcemaps in production
  target: ['chrome89'],
  loader: {
    '.js': 'jsx',
  },
  define: {
    'process.env.NODE_ENV': '"production"',
  },
  external: ['chrome'],
}).then(() => {
  console.log('JS build complete');
  
  // Copy HTML and CSS files
  fs.copyFileSync('popup/popup.html', 'dist-prod/popup/popup.html');
  fs.copyFileSync('popup/popup.css', 'dist-prod/popup/popup.css');
  fs.copyFileSync('sidepanel/sidepanel.html', 'dist-prod/sidepanel/sidepanel.html');
  fs.copyFileSync('sidepanel/sidepanel.css', 'dist-prod/sidepanel/sidepanel.css');
  
  // Copy shared utilities
  fs.copyFileSync('shared/whisper-utils.js', 'dist-prod/shared/whisper-utils.js');
  
  // Copy manifest with production settings
  const manifest = JSON.parse(fs.readFileSync('manifest.json', 'utf8'));
  manifest.name = "Whisper Transcribe"; // Remove (Dev) suffix
  manifest.permissions.push("microphone"); // Add microphone permission
  fs.writeFileSync('dist-prod/manifest.json', JSON.stringify(manifest, null, 2));
  
  // Copy assets
  if (fs.existsSync('assets/icon16.png')) {
    fs.copyFileSync('assets/icon16.png', 'dist-prod/assets/icon16.png');
  }
  if (fs.existsSync('assets/icon48.png')) {
    fs.copyFileSync('assets/icon48.png', 'dist-prod/assets/icon48.png');
  }
  if (fs.existsSync('assets/icon128.png')) {
    fs.copyFileSync('assets/icon128.png', 'dist-prod/assets/icon128.png');
  }
  
  console.log('Production build complete!');
}).catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
}); 