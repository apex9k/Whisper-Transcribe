import { build } from 'esbuild';
import fs from 'fs';
import path from 'path';

// Ensure dist directory exists
if (!fs.existsSync('dist')) {
  fs.mkdirSync('dist');
}

// Ensure dist/popup directory exists
if (!fs.existsSync('dist/popup')) {
  fs.mkdirSync('dist/popup');
}

// Ensure dist/background directory exists
if (!fs.existsSync('dist/background')) {
  fs.mkdirSync('dist/background');
}

// Ensure dist/assets directory exists
if (!fs.existsSync('dist/assets')) {
  fs.mkdirSync('dist/assets');
}

// Ensure dist/models directory exists
if (!fs.existsSync('dist/models')) {
  fs.mkdirSync('dist/models');
}

// Create a placeholder file in the models directory to ensure it's included in the extension
fs.writeFileSync('dist/models/.placeholder', 'This directory is used to store model files.');

// Build the JS files
build({
  entryPoints: ['popup/popup.js', 'background/background.js'],
  bundle: true,
  format: 'esm',
  outdir: 'dist',
  minify: true,
  sourcemap: true,
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
  fs.copyFileSync('popup/popup.html', 'dist/popup/popup.html');
  fs.copyFileSync('popup/popup.css', 'dist/popup/popup.css');
  
  // Copy manifest
  fs.copyFileSync('manifest.json', 'dist/manifest.json');
  
  // Copy assets if they exist
  if (fs.existsSync('assets/icon16.png')) {
    fs.copyFileSync('assets/icon16.png', 'dist/assets/icon16.png');
  }
  if (fs.existsSync('assets/icon48.png')) {
    fs.copyFileSync('assets/icon48.png', 'dist/assets/icon48.png');
  }
  if (fs.existsSync('assets/icon128.png')) {
    fs.copyFileSync('assets/icon128.png', 'dist/assets/icon128.png');
  }
  
  console.log('Build complete!');
}).catch((err) => {
  console.error('Build failed:', err);
  process.exit(1);
}); 