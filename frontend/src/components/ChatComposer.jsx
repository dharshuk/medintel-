// Input surface that merges text, uploads, and microphone control.
import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';

function ChatComposer({
  onSend,
  onUpload,
  onMicToggle,
  isListening,
  isThinking,
  transcript,
  onClearTranscript,
  sttSupported,
}) {
  const [input, setInput] = useState('');
  const fileInputRef = useRef(null);

  useEffect(() => {
    if (!transcript) return;
    setInput((prev) => `${prev ? `${prev} ` : ''}${transcript}`.trim());
  }, [transcript]);

  const handleSubmit = (event) => {
    event.preventDefault();
    onSend(input);
    setInput('');
    onClearTranscript();
  };

  return (
    <form onSubmit={handleSubmit} className="mt-4 rounded-3xl border border-white/10 bg-white/5 p-4 backdrop-blur-2xl">
      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="rounded-2xl border border-white/20 px-4 py-3 text-sm text-white/80 hover:border-neon hover:text-neon"
        >
          ⬆ Upload
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,image/*,audio/*"
          className="hidden"
          onChange={(event) => {
            onUpload(event.target.files);
            event.target.value = '';
          }}
        />

        <motion.button
          type="button"
          onClick={onMicToggle}
          whileTap={{ scale: 0.95 }}
          className={`rounded-2xl border px-4 py-3 text-sm transition ${
            isListening ? 'border-neon bg-neon/10 text-neon' : 'border-white/20 text-white/70'
          }`}
          disabled={!sttSupported}
        >
          {isListening ? '● Listening…' : sttSupported ? '🎙️ Voice' : 'Mic unavailable'}
        </motion.button>

        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Describe symptoms, upload labs, or ask for study tips…"
          className="h-16 flex-1 resize-none rounded-2xl border border-white/10 bg-black/30 px-4 py-3 text-white placeholder-white/50 focus:border-neon focus:outline-none"
        />

        <motion.button
          type="submit"
          whileTap={{ scale: 0.97 }}
          disabled={!input.trim() || isThinking}
          className="rounded-2xl bg-neon/90 px-6 py-3 font-semibold text-black transition disabled:cursor-not-allowed disabled:bg-white/20"
        >
          Send
        </motion.button>
      </div>
    </form>
  );
}

export default ChatComposer;
