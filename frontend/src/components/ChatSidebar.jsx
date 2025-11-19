// Sidebar with mode selector and session history tiles.
import { motion } from 'framer-motion';

function ChatSidebar({
  modes,
  history,
  activeMode,
  onModeChange,
  selectedHistory,
  onSelectHistory,
}) {
  return (
    <div className="flex h-full flex-col gap-6 rounded-3xl border border-white/5 bg-white/5 p-4 backdrop-blur-xl">
      <div>
        <p className="text-xs uppercase tracking-[0.4em] text-white/60">Modes</p>
        <div className="mt-3 space-y-3">
          {modes.map((mode) => {
            const isActive = activeMode === mode.id;
            return (
              <motion.button
                key={mode.id}
                layout
                onClick={() => onModeChange(mode.id)}
                className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                  isActive
                    ? 'border-white/40 bg-white/10 shadow-glow'
                    : 'border-white/10 bg-white/5 hover:border-white/30'
                }`}
                style={{ boxShadow: isActive ? `0 0 15px ${mode.accent}33` : 'none' }}
              >
                <p className="font-semibold" style={{ color: mode.accent }}>
                  {mode.label}
                </p>
                <p className="text-xs text-white/70">{mode.description}</p>
              </motion.button>
            );
          })}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <p className="text-xs uppercase tracking-[0.4em] text-white/60">Recent chats</p>
        <div className="mt-3 space-y-2">
          {history.map((item) => {
            const isActive = selectedHistory?.id === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectHistory(item)}
                className={`w-full rounded-2xl border px-4 py-3 text-left transition ${
                  isActive ? 'border-neon/60 bg-neon/5 shadow-glow' : 'border-white/10 bg-white/5'
                }`}
              >
                <p className="text-sm font-semibold text-white">{item.title}</p>
                <p className="text-xs text-white/60">{item.lastMessage}</p>
                <p className="mt-1 text-[10px] uppercase tracking-widest text-white/50">{item.timestamp}</p>
              </button>
            );
          })}
        </div>
      </div>

      <div className="rounded-2xl border border-neon/40 bg-neon/10 px-4 py-3 text-center text-sm text-white">
        Secure HIPAA-ready logging coming soon.
      </div>
    </div>
  );
}

export default ChatSidebar;
