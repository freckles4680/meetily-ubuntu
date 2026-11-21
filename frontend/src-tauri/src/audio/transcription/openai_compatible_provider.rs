// audio/transcription/openai_compatible_provider.rs
//
// Transcription provider for any server that implements OpenAI's
// `POST {base_url}/audio/transcriptions` endpoint: OpenAI itself, or a local
// server such as llama.cpp (e.g. Voxtral), vLLM, or faster-whisper-server.
//
// Audio is sent as a 16 kHz mono 16-bit PCM WAV file per VAD speech chunk.
// When the base URL points at a remote host, meeting audio leaves this machine.

use super::provider::{TranscriptionError, TranscriptionProvider, TranscriptResult};
use async_trait::async_trait;
use log::{info, warn};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Provider id stored in `transcript_settings.provider`.
pub const OPENAI_COMPATIBLE_PROVIDER_ID: &str = "custom-openai";

const SAMPLE_RATE: u32 = 16_000;
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Saved configuration for the OpenAI-compatible transcription provider.
/// Stored as JSON in `transcript_settings.customOpenAIConfig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptCustomOpenAIConfig {
    /// Base URL of the API, including the version prefix (e.g. "https://api.openai.com/v1")
    pub endpoint: String,
    /// API key (optional for local servers that don't check it)
    #[serde(rename = "apiKey")]
    pub api_key: Option<String>,
    /// Model identifier (e.g. "whisper-1", "gpt-4o-transcribe", "voxtral-mini")
    pub model: String,
}

impl TranscriptCustomOpenAIConfig {
    /// Validate and normalize user input. Returns a user-facing error message on failure.
    pub fn validated(endpoint: &str, api_key: Option<String>, model: &str) -> Result<Self, String> {
        let endpoint = endpoint.trim().trim_end_matches('/').to_string();
        let model = model.trim().to_string();
        if endpoint.is_empty() {
            return Err("Base URL is required".to_string());
        }
        if !endpoint.starts_with("http://") && !endpoint.starts_with("https://") {
            return Err("Base URL must start with http:// or https://".to_string());
        }
        if url::Url::parse(&endpoint).is_err() {
            return Err("Base URL is not a valid URL".to_string());
        }
        if model.is_empty() {
            return Err("Model name is required".to_string());
        }
        Ok(Self {
            endpoint,
            api_key: api_key
                .map(|k| k.trim().to_string())
                .filter(|k| !k.is_empty()),
            model,
        })
    }

    pub fn transcriptions_url(&self) -> String {
        format!("{}/audio/transcriptions", self.endpoint.trim_end_matches('/'))
    }
}

/// Map Meetily's language preference to the `language` form field.
/// "auto" / "auto-translate" / empty mean "let the server detect it".
fn language_field(language: Option<&str>) -> Option<String> {
    match language.map(str::trim) {
        None | Some("") | Some("auto") | Some("auto-translate") => None,
        Some(lang) => Some(lang.to_string()),
    }
}

/// Encode 16 kHz mono f32 samples as a 16-bit PCM WAV file.
pub fn encode_wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut wav = Vec::with_capacity(44 + data_len as usize);
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // PCM fmt chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&1u16.to_le_bytes()); // mono
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    wav.extend_from_slice(&2u16.to_le_bytes()); // block align
    wav.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    for &sample in samples {
        let value = (sample.clamp(-1.0, 1.0) * i16::MAX as f32) as i16;
        wav.extend_from_slice(&value.to_le_bytes());
    }
    wav
}

/// Remove the markup some ASR models leave in their text output.
/// Qwen3-ASR served by llama.cpp returns e.g. "language English<asr_text>Hello there"
/// (and "language None<asr_text>" for silence); a closing "</asr_text>" may also appear.
pub fn strip_asr_markup(text: &str) -> String {
    static PREFIX: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
        regex::Regex::new(r"(?i)(?:language\s+[^<\n]{0,40})?<asr_text>|</asr_text>")
            .expect("valid regex")
    });
    if !text.contains("asr_text>") {
        return text.to_string();
    }
    PREFIX.replace_all(text, " ").split_whitespace().collect::<Vec<_>>().join(" ")
}

#[derive(Deserialize)]
struct TranscriptionResponse {
    text: String,
}

/// Send one transcription request. Shared by the provider and the settings "Test" button.
pub async fn request_transcription(
    client: &reqwest::Client,
    config: &TranscriptCustomOpenAIConfig,
    samples: &[f32],
    language: Option<&str>,
) -> Result<String, TranscriptionError> {
    let file_part = reqwest::multipart::Part::bytes(encode_wav_pcm16(samples, SAMPLE_RATE))
        .file_name("audio.wav")
        .mime_str("audio/wav")
        .map_err(|e| TranscriptionError::EngineFailed(e.to_string()))?;

    let mut form = reqwest::multipart::Form::new()
        .part("file", file_part)
        .text("model", config.model.clone())
        .text("response_format", "json");
    if let Some(lang) = language_field(language) {
        form = form.text("language", lang);
    }

    let mut request = client.post(config.transcriptions_url()).multipart(form);
    if let Some(key) = &config.api_key {
        request = request.bearer_auth(key);
    }

    let origin = crate::utils::url_origin_for_log(&config.endpoint);
    let response = request.send().await.map_err(|e| {
        let class = if e.is_timeout() {
            "timed out"
        } else if e.is_connect() {
            "could not connect"
        } else {
            "request failed"
        };
        warn!("OpenAI-compatible transcription {}; endpoint_origin={}", class, origin);
        TranscriptionError::EngineFailed(format!("Transcription server {} ({})", class, origin))
    })?;

    let status = response.status();
    if !status.is_success() {
        // Error bodies from these APIs describe the request, not the audio, but keep them short.
        let body = response.text().await.unwrap_or_default();
        let snippet: String = body.chars().take(300).collect();
        warn!(
            "OpenAI-compatible transcription failed; endpoint_origin={}, status={}",
            origin, status
        );
        let hint = match status.as_u16() {
            401 | 403 => " Check the API key.",
            404 => " Check the base URL (it usually ends in /v1) and the model name.",
            _ => "",
        };
        return Err(TranscriptionError::EngineFailed(format!(
            "Transcription server returned {}.{} {}",
            status, hint, snippet
        )));
    }

    let parsed: TranscriptionResponse = response.json().await.map_err(|_| {
        TranscriptionError::EngineFailed(
            "Transcription server response is missing the 'text' field; is it OpenAI-compatible?"
                .to_string(),
        )
    })?;
    Ok(strip_asr_markup(&parsed.text))
}

pub fn build_http_client() -> Result<reqwest::Client, String> {
    reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .connect_timeout(CONNECT_TIMEOUT)
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))
}

/// Transcription provider backed by an OpenAI-compatible HTTP API.
pub struct OpenAICompatibleProvider {
    config: TranscriptCustomOpenAIConfig,
    client: reqwest::Client,
}

impl OpenAICompatibleProvider {
    pub fn new(config: TranscriptCustomOpenAIConfig) -> Result<Self, String> {
        info!(
            "Using OpenAI-compatible transcription; endpoint_origin={}, model={}",
            crate::utils::url_origin_for_log(&config.endpoint),
            config.model
        );
        Ok(Self {
            config,
            client: build_http_client()?,
        })
    }
}

#[async_trait]
impl TranscriptionProvider for OpenAICompatibleProvider {
    async fn transcribe(
        &self,
        audio: Vec<f32>,
        language: Option<String>,
    ) -> std::result::Result<TranscriptResult, TranscriptionError> {
        // Very short chunks are rejected by some servers and are rarely real speech.
        const MIN_SAMPLES: usize = (SAMPLE_RATE as usize) / 10; // 100 ms
        if audio.len() < MIN_SAMPLES {
            return Err(TranscriptionError::AudioTooShort {
                samples: audio.len(),
                minimum: MIN_SAMPLES,
            });
        }

        let text = request_transcription(&self.client, &self.config, &audio, language.as_deref())
            .await?;
        Ok(TranscriptResult {
            text: text.trim().to_string(),
            confidence: None,
            is_partial: false,
        })
    }

    async fn is_model_loaded(&self) -> bool {
        // Nothing to load locally; the server owns the model.
        true
    }

    async fn get_current_model(&self) -> Option<String> {
        Some(self.config.model.clone())
    }

    fn provider_name(&self) -> &'static str {
        "OpenAI-compatible API"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_is_valid_pcm16_mono() {
        let wav = encode_wav_pcm16(&[0.0, 1.0, -1.0, 0.5], 16_000);
        assert_eq!(wav.len(), 44 + 8);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 8);
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(i16::from_le_bytes(wav[46..48].try_into().unwrap()), i16::MAX);
        assert_eq!(i16::from_le_bytes(wav[48..50].try_into().unwrap()), -i16::MAX);
    }

    #[test]
    fn strips_qwen3_asr_markup() {
        assert_eq!(
            strip_asr_markup("language English<asr_text>And God, we pray."),
            "And God, we pray."
        );
        assert_eq!(strip_asr_markup("language None<asr_text>"), "");
        assert_eq!(strip_asr_markup("<asr_text>Hi</asr_text>"), "Hi");
        assert_eq!(
            strip_asr_markup("language English<asr_text>One. language English<asr_text>Two."),
            "One. Two."
        );
        // Ordinary text, including the word "language", is untouched
        assert_eq!(
            strip_asr_markup("The language  of  the meeting."),
            "The language  of  the meeting."
        );
    }

    #[test]
    fn language_field_skips_auto_modes() {
        assert_eq!(language_field(None), None);
        assert_eq!(language_field(Some("auto")), None);
        assert_eq!(language_field(Some("auto-translate")), None);
        assert_eq!(language_field(Some("de")), Some("de".to_string()));
    }

    #[test]
    fn config_validation_normalizes_input() {
        let config = TranscriptCustomOpenAIConfig::validated(
            " http://localhost:8080/v1/ ",
            Some("  ".to_string()),
            " voxtral ",
        )
        .unwrap();
        assert_eq!(config.endpoint, "http://localhost:8080/v1");
        assert_eq!(config.api_key, None);
        assert_eq!(config.model, "voxtral");
        assert_eq!(
            config.transcriptions_url(),
            "http://localhost:8080/v1/audio/transcriptions"
        );

        assert!(TranscriptCustomOpenAIConfig::validated("localhost:8080", None, "m").is_err());
        assert!(TranscriptCustomOpenAIConfig::validated("http://x/v1", None, " ").is_err());
    }
}
