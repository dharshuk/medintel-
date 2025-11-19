// Animated AI avatar with status indicators and glow effects

import { motion } from 'framer-motion';

const modeConfig = {
  medical: { label: 'Clinical Mode', color: '#23c68b', icon: '🩺' },
  general: { label: 'General Mode', color: '#28F7CE', icon: '💬' },
  mental: { label: 'Mental Health Mode', color: '#C084FC', icon: '🧘' },
  student: { label: 'Student Mode', color: '#FFB800', icon: '🎓' },
};

function Avatar({ mode, isThinking, isSpeaking, isListening }) {
  const config = modeConfig[mode] || modeConfig.medical;
  
  const getStatus = () => {
    if (isThinking) return { text: 'Analyzing...', color: '#28F7CE' };
    if (isSpeaking) return { text: 'Speaking...', color: '#FFB800' };
    if (isListening) return { text: 'Listening...', color: '#FF4D6D' };
    return { text: 'Ready', color: config.color };
  };

  const status = getStatus();
  const isActive = isThinking || isSpeaking || isListening;

  return (
    <div className="flex items-center gap-4">
      {/* Avatar Circle */}
      <div className="relative">
        {/* Outer glow rings */}
        {isActive && (
          <>
            <motion.div
              className="absolute inset-0 rounded-full"
              style={{ backgroundColor: status.color, opacity: 0.2 }}
              animate={{ scale: [1, 1.4, 1], opacity: [0.2, 0, 0.2] }}
              transition={{ repeat: Infinity, duration: 2 }}
            />
            <motion.div
              className="absolute inset-0 rounded-full"
              style={{ backgroundColor: status.color, opacity: 0.3 }}
              animate={{ scale: [1, 1.2, 1], opacity: [0.3, 0, 0.3] }}
              transition={{ repeat: Infinity, duration: 2, delay: 0.4 }}
            />
          </>
        )}

        {/* Main avatar */}
        <motion.div
          className="relative flex h-16 w-16 items-center justify-center rounded-full border-2 border-white/20 bg-gradient-to-br from-bg-navy to-bg-dark shadow-glow"
          style={{ borderColor: status.color }}
          animate={
            isActive
              ? { boxShadow: [`0 0 20px ${status.color}`, `0 0 40px ${status.color}`, `0 0 20px ${status.color}`] }
              : {}
          }
          transition={{ repeat: Infinity, duration: 2 }}
        >
          <motion.div
            className="text-3xl"
            animate={isActive ? { scale: [1, 1.1, 1] } : {}}
            transition={{ repeat: Infinity, duration: 1.5 }}
          >
            {config.icon}
          </motion.div>
        </motion.div>

        {/* Status indicator dot */}
        <motion.div
          className="absolute bottom-0 right-0 h-4 w-4 rounded-full border-2 border-bg-dark"
          style={{ backgroundColor: status.color }}
          animate={isActive ? { scale: [1, 1.2, 1] } : {}}
          transition={{ repeat: Infinity, duration: 1 }}
        />
      </div>

      {/* Info */}
      <div>
        <div className="flex items-center gap-2">
          <p className="text-sm font-semibold text-white">{config.label}</p>
          <span
            className="h-2 w-2 rounded-full"
            style={{ backgroundColor: status.color }}
          />
        </div>
        <p className="text-xs" style={{ color: status.color }}>
          {status.text}
        </p>
      </div>
    </div>
  );
}

export default Avatar;
