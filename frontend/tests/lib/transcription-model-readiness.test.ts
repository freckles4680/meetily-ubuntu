import { describe, expect, test } from 'bun:test';

import {
  getProviderCommands,
  hasDownloadingModel,
  isApiTranscriptionProvider,
  isLoopbackEndpoint,
  isTranscriptApiConfigReady,
} from '../../src/lib/transcription-model-readiness';

describe('transcription model readiness', () => {
  test('uses Whisper commands when Whisper is the configured provider', () => {
    expect(getProviderCommands('localWhisper')).toEqual({
      initialize: 'whisper_init',
      hasAvailableModels: 'whisper_has_available_models',
      getAvailableModels: 'whisper_get_available_models',
    });
  });

  test('uses Parakeet commands when Parakeet is the configured provider', () => {
    expect(getProviderCommands('parakeet')).toEqual({
      initialize: 'parakeet_init',
      hasAvailableModels: 'parakeet_has_available_models',
      getAvailableModels: 'parakeet_get_available_models',
    });
  });

  test('does not silently treat an unsupported provider as Parakeet', () => {
    expect(getProviderCommands('deepgram')).toBeNull();
  });

  test('recognizes only active downloads', () => {
    expect(hasDownloadingModel([{ status: 'Available' }])).toBeFalse();
    expect(hasDownloadingModel([{ status: 'Downloading' }])).toBeTrue();
    expect(hasDownloadingModel([{ status: { Downloading: { progress: 0 } } }])).toBeTrue();
    expect(hasDownloadingModel([{ status: { Downloading: { progress: 42 } } }])).toBeTrue();
  });
});

describe('OpenAI-compatible transcription provider', () => {
  test('is treated as an API provider without local model commands', () => {
    expect(isApiTranscriptionProvider('custom-openai')).toBeTrue();
    expect(isApiTranscriptionProvider('parakeet')).toBeFalse();
    expect(getProviderCommands('custom-openai')).toBeNull();
  });

  test('is ready only with a base URL and model', () => {
    expect(isTranscriptApiConfigReady(null)).toBeFalse();
    expect(isTranscriptApiConfigReady({ endpoint: 'http://localhost:8080/v1', model: ' ' })).toBeFalse();
    expect(isTranscriptApiConfigReady({ endpoint: 'http://localhost:8080/v1', model: 'voxtral' })).toBeTrue();
  });

  test('flags only loopback URLs as staying on this machine', () => {
    expect(isLoopbackEndpoint('http://localhost:8080/v1')).toBeTrue();
    expect(isLoopbackEndpoint('http://127.0.0.1:8000/v1')).toBeTrue();
    expect(isLoopbackEndpoint('http://[::1]:8000/v1')).toBeTrue();
    expect(isLoopbackEndpoint('https://api.openai.com/v1')).toBeFalse();
    expect(isLoopbackEndpoint('http://192.168.1.20:8000/v1')).toBeFalse();
    expect(isLoopbackEndpoint('http://localhost.evil.com/v1')).toBeFalse();
    expect(isLoopbackEndpoint('not a url')).toBeFalse();
  });
});
