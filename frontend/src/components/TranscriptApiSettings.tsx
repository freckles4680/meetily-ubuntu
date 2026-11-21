import { useEffect, useState } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { toast } from 'sonner';
import { AlertTriangle, Eye, EyeOff, Loader2, ShieldCheck } from 'lucide-react';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Label } from './ui/label';
import {
  isLoopbackEndpoint,
  isTranscriptApiConfigReady,
  type TranscriptApiConfig,
} from '@/lib/transcription-model-readiness';

interface TranscriptApiSettingsProps {
  /** Called after the settings are saved and selected as the active provider. */
  onSaved: (config: { model: string }) => void;
}

/**
 * Settings for transcription through any OpenAI-compatible
 * `POST {baseUrl}/audio/transcriptions` endpoint (OpenAI, llama.cpp, vLLM, ...).
 */
export function TranscriptApiSettings({ onSaved }: TranscriptApiSettingsProps) {
  const [endpoint, setEndpoint] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [model, setModel] = useState('');
  const [showApiKey, setShowApiKey] = useState(false);
  const [isTesting, setIsTesting] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    invoke<TranscriptApiConfig | null>('api_get_transcript_custom_openai_config')
      .then((config) => {
        if (config) {
          setEndpoint(config.endpoint || '');
          setApiKey(config.apiKey || '');
          setModel(config.model || '');
        }
      })
      .catch((error) => console.error('Failed to load transcription API settings:', error));
  }, []);

  const trimmedEndpoint = endpoint.trim();
  const canSubmit = isTranscriptApiConfigReady({ endpoint, model });
  const isLocal = trimmedEndpoint !== '' && isLoopbackEndpoint(trimmedEndpoint);
  const args = { endpoint: trimmedEndpoint, apiKey: apiKey.trim() || null, model: model.trim() };

  const handleTest = async () => {
    setIsTesting(true);
    try {
      const result = await invoke<{ message: string }>('api_test_transcript_custom_openai_connection', args);
      toast.success('Connection successful', { description: result.message });
    } catch (error) {
      toast.error('Connection failed', { description: String(error) });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await invoke('api_save_transcript_custom_openai_config', args);
      toast.success('Transcription API saved', {
        description: `New recordings will be transcribed with ${args.model}.`,
      });
      onSaved({ model: args.model });
    } catch (error) {
      toast.error('Failed to save transcription API settings', { description: String(error) });
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="mt-6 space-y-4 mx-1">
      <div>
        <Label className="block text-sm font-medium text-gray-700 mb-1">Base URL</Label>
        <Input
          value={endpoint}
          onChange={(e) => setEndpoint(e.target.value)}
          placeholder="http://localhost:8080/v1"
          className="focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
        />
        <p className="mt-1 text-xs text-gray-500">
          Meetily calls <code>{'{base URL}'}/audio/transcriptions</code>. Use{' '}
          <code>https://api.openai.com/v1</code> for OpenAI, or a local server such as llama.cpp
          (e.g. Voxtral), vLLM, or faster-whisper-server.
        </p>
      </div>

      <div>
        <Label className="block text-sm font-medium text-gray-700 mb-1">Model</Label>
        <Input
          value={model}
          onChange={(e) => setModel(e.target.value)}
          placeholder="whisper-1, gpt-4o-transcribe, voxtral-mini..."
          className="focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
        />
      </div>

      <div>
        <Label className="block text-sm font-medium text-gray-700 mb-1">API Key (optional)</Label>
        <div className="relative">
          <Input
            type={showApiKey ? 'text' : 'password'}
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Leave empty if your server doesn't need one"
            className="pr-10 focus:ring-1 focus:ring-blue-500 focus:border-blue-500"
          />
          <div className="absolute inset-y-0 right-0 pr-1 flex items-center">
            <Button type="button" variant="ghost" size="icon" onClick={() => setShowApiKey(!showApiKey)}>
              {showApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </Button>
          </div>
        </div>
      </div>

      {trimmedEndpoint !== '' && (
        isLocal ? (
          <div className="flex items-start gap-2 rounded-md border border-green-200 bg-green-50 p-3 text-sm text-green-800">
            <ShieldCheck className="h-4 w-4 mt-0.5 shrink-0" />
            <span>This URL points at this computer, so meeting audio stays on your machine.</span>
          </div>
        ) : (
          <div className="flex items-start gap-2 rounded-md border border-amber-300 bg-amber-50 p-3 text-sm text-amber-900">
            <AlertTriangle className="h-4 w-4 mt-0.5 shrink-0" />
            <span>
              Meeting audio will be sent to <strong>{safeHost(trimmedEndpoint)}</strong> while you record.
              It leaves this computer and is handled under that service&apos;s privacy terms.
            </span>
          </div>
        )
      )}

      <div className="flex gap-2">
        <Button type="button" variant="outline" onClick={handleTest} disabled={!canSubmit || isTesting || isSaving}>
          {isTesting && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Test connection
        </Button>
        <Button type="button" onClick={handleSave} disabled={!canSubmit || isTesting || isSaving}>
          {isSaving && <Loader2 className="h-4 w-4 mr-2 animate-spin" />}
          Save and use
        </Button>
      </div>
    </div>
  );
}

function safeHost(endpoint: string): string {
  try {
    return new URL(endpoint).host;
  } catch {
    return endpoint;
  }
}
