-- Migration: OpenAI-compatible transcription provider ("custom-openai")
-- This column stores: {endpoint, apiKey, model}
ALTER TABLE transcript_settings ADD COLUMN customOpenAIConfig TEXT;
