#!/usr/bin/env node
/**
 * Auto-detect GPU and run Tauri with appropriate features
 */

const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Get the command (dev or build)
const command = process.argv[2];
if (!command || !['dev', 'build'].includes(command)) {
  console.error('Usage: node tauri-auto.js [dev|build]');
  process.exit(1);
}

// Detect GPU feature
let feature = '';

// Check for environment variable override first
if (process.env.TAURI_GPU_FEATURE) {
  feature = process.env.TAURI_GPU_FEATURE;
  console.log(`🔧 Using forced GPU feature from environment: ${feature}`);
} else {
  try {
    const result = execSync('node scripts/auto-detect-gpu.js', {
      encoding: 'utf8',
      stdio: ['pipe', 'pipe', 'inherit']
    });
    feature = result.trim();
  } catch (err) {
    // If detection fails, continue with no features
  }
}

console.log(''); // Empty line for spacing

// Platform-specific environment variables
const platform = os.platform();
const env = { ...process.env };

if (platform === 'linux' && feature === 'cuda') {
  console.log('🐧 Linux/CUDA detected: Setting CMAKE flags for NVIDIA GPU');
  env.CMAKE_CUDA_ARCHITECTURES = process.env.CMAKE_CUDA_ARCHITECTURES || '75';
  env.CMAKE_CUDA_STANDARD = '17';
  env.CMAKE_POSITION_INDEPENDENT_CODE = 'ON';
}

// Build the tauri command
let tauriCmd = `tauri ${command}`;

// tauri.conf.json sets createUpdaterArtifacts, which makes `tauri build` fail after
// bundling when no updater signing key is available (local builds, forks).
// Without a key, skip the updater artifacts so the build finishes cleanly.
// MEETILY_TAURI_CONFIG may point to an extra JSON config to merge in (the CUDA
// package build uses it to bundle the CUDA runtime libraries into the .deb).
let override = {};
if (process.env.MEETILY_TAURI_CONFIG) {
  override = JSON.parse(fs.readFileSync(process.env.MEETILY_TAURI_CONFIG, 'utf8'));
  console.log(`🔧 Merging extra Tauri config from ${process.env.MEETILY_TAURI_CONFIG}`);
}
if (command === 'build' && !process.env.TAURI_SIGNING_PRIVATE_KEY) {
  override.bundle = { ...(override.bundle || {}), createUpdaterArtifacts: false };
  console.log('ℹ️  TAURI_SIGNING_PRIVATE_KEY not set: building without signed updater artifacts');
}
if (Object.keys(override).length > 0) {
  const overridePath = path.join(os.tmpdir(), 'meetily-tauri-override.json');
  fs.writeFileSync(overridePath, JSON.stringify(override));
  tauriCmd += ` --config "${overridePath}"`;
}
if (feature && feature !== 'none') {
  tauriCmd += ` -- --features ${feature}`;
  console.log(`🚀 Running: tauri ${command} with features: ${feature}`);
} else {
  console.log(`🚀 Running: tauri ${command} (CPU-only mode)`);
}
console.log('');

// Execute the command
try {
  execSync(tauriCmd, { stdio: 'inherit', env });
} catch (err) {
  process.exit(err.status || 1);
}
