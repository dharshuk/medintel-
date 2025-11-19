// Enhanced chat bubble with risk badges, citations, and TTS playback

import { motion } from 'framer-motion';

const riskColors = {
  Green: 'risk-badge-green',
  Amber: 'risk-badge-amber',
  Red: 'risk-badge-red',
};

function ChatBubble({ message, onPlayAudio, onStopAudio, showAudio, speaking }) {
  const isAssistant = message.role === 'assistant';
  const riskBadge = isAssistant && message.riskLevel ? riskColors[message.riskLevel] : null;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, translateY: 20 }}
      animate={{ opacity: 1, translateY: 0 }}
      transition={{ duration: 0.3 }}
      className={`glass-card p-5 ${
        isAssistant ? 'ml-0 mr-12' : 'ml-12 mr-0 border-primary/20 bg-primary/5'
      }`}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <p className="text-xs font-semibold uppercase tracking-wider text-white/60">
            {isAssistant ? '🩺 MedIntel' : '👤 You'}
          </p>
          {message.timestamp && (
            <span className="text-xs text-white/40">
              {new Date(message.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
        {riskBadge && <span className={riskBadge}>{message.riskLevel}</span>}
      </div>

      {message.summary && (
        <p className="mb-2 text-sm font-semibold text-primary">{message.summary}</p>
      )}

      <div className="prose prose-invert max-w-none text-base leading-relaxed text-white/90">
        {message.content}
      </div>

      {message.nextSteps && message.nextSteps.length > 0 && (
        <div className="mt-4 rounded-xl border border-primary/20 bg-primary/5 p-3">
          <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-primary">
            Next Steps
          </p>
          <ul className="space-y-1.5 text-sm text-white/80">
            {message.nextSteps.map((step, idx) => (
              <li key={idx} className="flex items-start gap-2">
                <span className="text-primary">→</span>
                <span>{step}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {message.sources && message.sources.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-2">
          {message.sources.map((source, idx) => (
            <span
              key={idx}
              className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/60"
            >
              📚 {source}
            </span>
          ))}
        </div>
      )}

      {message.confidence && (
        <p className="mt-2 text-xs text-white/40">Confidence: {message.confidence}</p>
      )}

      {showAudio && isAssistant && message.content && (
        <div className="mt-3 flex gap-2">
          {!speaking ? (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => onPlayAudio(message.content)}
              className="flex items-center gap-2 rounded-xl border border-primary/30 bg-primary/10 px-4 py-2 text-sm text-primary transition-all hover:bg-primary/20"
            >
              <span>▶</span>
              <span>Play Audio</span>
            </motion.button>
          ) : (
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={onStopAudio}
              className="flex items-center gap-2 rounded-xl border border-red-500/30 bg-red-500/10 px-4 py-2 text-sm text-red-400 transition-all hover:bg-red-500/20"
            >
              <span>⏹</span>
              <span>Stop Audio</span>
            </motion.button>
          )}
        </div>
      )}
    </motion.div>
  );
}

export default ChatBubble;
