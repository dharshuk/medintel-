// Enhanced composer with text input, voice, upload, and quick prompts

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { quickPrompts, tonePresets } from '../data/dummyData.js';

function Composer({
  onSend,
  onUpload,
  onMicToggle,
  isListening,
  isThinking,
  transcript,
  interimTranscript,
  onClearTranscript,
  sttSupported,
}) {
  const [input, setInput] = useState('');
  const [showPresets, setShowPresets] = useState(false);
  const [selectedTone, setSelectedTone] = useState('detailed');
  const fileInputRef = useRef(null);
  const textareaRef = useRef(null);

  // Update input with transcript
  useEffect(() => {
    if (transcript) {
      setInput((prev) => {
        const combined = prev ? `${prev} ${transcript}` : transcript;
        return combined.trim();
      });
      onClearTranscript();
    }
  }, [transcript, onClearTranscript]);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${textareaRef.current.scrollHeight}px`;
    }
  }, [input]);

  const handleSubmit = (e) => {
    e?.preventDefault();
    if (!input.trim() || isThinking) return;

    onSend(input);
    setInput('');
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleQuickPrompt = (promptText) => {
    setInput(promptText);
    setShowPresets(false);
  };

  return (
    <div className="space-y-3">
      {/* Quick Prompts */}
      {showPresets && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex flex-wrap gap-2"
        >
          {quickPrompts.map((prompt) => (
            <button
              key={prompt.id}
              onClick={() => handleQuickPrompt(prompt.text)}
              className="glass-card-hover flex items-center gap-2 rounded-xl px-3 py-2 text-sm text-white/80"
            >
              <span>{prompt.icon}</span>
              <span>{prompt.text}</span>
            </button>
          ))}
        </motion.div>
      )}

      {/* Tone Presets */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-white/50">Tone:</span>
        <div className="flex gap-1">
          {tonePresets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => setSelectedTone(preset.id)}
              className={`rounded-lg px-3 py-1 text-xs transition-all ${
                selectedTone === preset.id
                  ? 'bg-primary/20 text-primary'
                  : 'bg-white/5 text-white/60 hover:bg-white/10'
              }`}
              title={preset.desc}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main Input Area */}
      <form onSubmit={handleSubmit} className="glass-card p-4">
        <div className="flex items-end gap-3">
          {/* Upload Button */}
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => fileInputRef.current?.click()}
            className="btn-secondary shrink-0"
            aria-label="Upload file"
          >
            📎
          </motion.button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".pdf,.png,.jpg,.jpeg,.mp3,.wav"
            className="hidden"
            onChange={(e) => {
              onUpload(e.target.files);
              e.target.value = '';
            }}
          />

          {/* Microphone Button */}
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={onMicToggle}
            className={`btn-secondary shrink-0 transition-all ${
              isListening
                ? 'animate-pulseGlow border-primary bg-primary/20 text-primary'
                : ''
            }`}
            disabled={!sttSupported}
            aria-label={isListening ? 'Stop listening' : 'Start voice input'}
          >
            {isListening ? '●' : '🎙️'}
          </motion.button>

          {/* Text Input */}
          <div className="relative flex-1">
            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Describe symptoms, upload labs, or ask for study tips..."
              className="input-field w-full resize-none"
              rows={1}
              style={{ minHeight: '48px', maxHeight: '200px' }}
            />
            {interimTranscript && (
              <div className="absolute bottom-full left-0 mb-1 rounded-lg bg-primary/10 px-2 py-1 text-xs text-primary">
                {interimTranscript}
              </div>
            )}
          </div>

          {/* Quick Prompts Toggle */}
          <motion.button
            type="button"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setShowPresets(!showPresets)}
            className={`btn-secondary shrink-0 ${showPresets ? 'bg-primary/20 text-primary' : ''}`}
            aria-label="Toggle quick prompts"
          >
            ⚡
          </motion.button>

          {/* Send Button */}
          <motion.button
            type="submit"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            disabled={!input.trim() || isThinking}
            className="btn-primary shrink-0"
          >
            {isThinking ? '⏳' : '➤'}
          </motion.button>
        </div>
      </form>
    </div>
  );
}

export default Composer;
