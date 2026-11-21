export interface ModelWithStatus {
  status?: unknown;
}

interface ProviderCommands {
  initialize: string;
  hasAvailableModels: string;
  getAvailableModels: string;
}

const PROVIDER_COMMANDS: Record<string, ProviderCommands> = {
  localWhisper: {
    initialize: 'whisper_init',
    hasAvailableModels: 'whisper_has_available_models',
    getAvailableModels: 'whisper_get_available_models',
  },
  parakeet: {
    initialize: 'parakeet_init',
    hasAvailableModels: 'parakeet_has_available_models',
    getAvailableModels: 'parakeet_get_available_models',
  },
};

export function getProviderCommands(provider: string): ProviderCommands | null {
  return PROVIDER_COMMANDS[provider] ?? null;
}

export function hasDownloadingModel(models: ModelWithStatus[]): boolean {
  return models.some(({ status }) => (
    status === 'Downloading'
    || (status !== null && typeof status === 'object' && 'Downloading' in status)
  ));
}

/** Provider id for transcription through an OpenAI-compatible /audio/transcriptions API. */
export const OPENAI_COMPATIBLE_TRANSCRIPTION_PROVIDER = 'custom-openai';

export interface TranscriptApiConfig {
  endpoint?: string | null;
  apiKey?: string | null;
  model?: string | null;
}

/** True for providers that send audio to an HTTP API instead of a local model. */
export function isApiTranscriptionProvider(provider: string): boolean {
  return provider === OPENAI_COMPATIBLE_TRANSCRIPTION_PROVIDER;
}

/** An API provider is ready once a base URL and model have been saved. */
export function isTranscriptApiConfigReady(config: TranscriptApiConfig | null | undefined): boolean {
  return Boolean(config?.endpoint?.trim() && config?.model?.trim());
}

/**
 * True when the base URL points at this machine, so audio stays local.
 * Anything else (including LAN addresses) is treated as leaving the machine.
 */
export function isLoopbackEndpoint(endpoint: string): boolean {
  try {
    const host = new URL(endpoint.trim()).hostname.toLowerCase();
    return host === 'localhost'
      || host.endsWith('.localhost')
      || host === '[::1]'
      || host === '::1'
      || /^127(\.\d{1,3}){3}$/.test(host);
  } catch {
    return false;
  }
}
