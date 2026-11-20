import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { Cpu, Zap } from 'lucide-react';

/** Mirrors WhisperRuntimeAcceleration in src-tauri/src/whisper_engine/acceleration.rs */
export interface WhisperAcceleration {
  backend: string;
  uses_gpu: boolean;
  device: string | null;
  label: string;
}

// The answer does not change while the app runs, so ask the backend once.
let accelerationPromise: Promise<WhisperAcceleration | null> | null = null;

export function getWhisperAcceleration(): Promise<WhisperAcceleration | null> {
  if (!accelerationPromise) {
    accelerationPromise = invoke<WhisperAcceleration>('whisper_get_acceleration').catch((error) => {
      console.error('Failed to get Whisper acceleration:', error);
      accelerationPromise = null;
      return null;
    });
  }
  return accelerationPromise;
}

interface TranscriptionHardwareLabelProps {
  /** Transcription provider: 'localWhisper' (or 'whisper'), 'parakeet', or a cloud provider. */
  provider: string | undefined | null;
  className?: string;
}

/** Shows whether the selected local transcription engine runs on the GPU or the CPU. */
export function TranscriptionHardwareLabel({ provider, className = '' }: TranscriptionHardwareLabelProps) {
  const [whisper, setWhisper] = useState<WhisperAcceleration | null>(null);
  const isWhisper = provider === 'localWhisper' || provider === 'whisper';

  useEffect(() => {
    if (!isWhisper) return;
    let active = true;
    getWhisperAcceleration().then((info) => {
      if (active) setWhisper(info);
    });
    return () => {
      active = false;
    };
  }, [isWhisper]);

  let usesGpu = false;
  let text: string | null = null;
  if (provider === 'parakeet') {
    text = 'CPU (Parakeet always runs on the CPU)';
  } else if (isWhisper && whisper) {
    usesGpu = whisper.uses_gpu;
    text = whisper.label;
  }
  if (!text) return null;

  const Icon = usesGpu ? Zap : Cpu;
  return (
    <p className={`flex items-center gap-1.5 text-xs text-gray-500 ${className}`}>
      <Icon className={`h-3.5 w-3.5 ${usesGpu ? 'text-green-600' : 'text-gray-400'}`} />
      <span>Runs on: {text}</span>
    </p>
  );
}
