// Main chat center with messages, avatar header, and composer

import { useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import ChatBubble from './ChatBubble.jsx';
import Composer from './Composer.jsx';
import Avatar from './Avatar.jsx';
import Waveform from './Waveform.jsx';

function ChatCenter({
  messages,
  isThinking,
  activeMode,
  onSendMessage,
  onFileUpload,
  onPlayAudio,
  onStopAudio,
  speaking,
  listening,
  onMicToggle,
  transcript,
  interimTranscript,
  onClearTranscript,
  sttSupported,
  ttsSupported,
}) {
  const messagesEndRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <section className="medintel-bg relative flex flex-1 flex-col overflow-hidden">
      {/* Avatar Header */}
      <div className="border-b border-white/5 p-4">
        <Avatar mode={activeMode} isThinking={isThinking} isSpeaking={speaking} isListening={listening} />
      </div>

      {/* Messages Container */}
      <div className="flex-1 space-y-4 overflow-y-auto p-6">
        {messages.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex h-full flex-col items-center justify-center text-center"
          >
            <div className="mb-4 text-6xl">🩺</div>
            <h2 className="mb-2 text-2xl font-bold text-white">Welcome to MedIntel</h2>
            <p className="max-w-md text-white/60">
              Your AI medical assistant. Ask questions, upload reports, or get step-by-step learning guidance.
            </p>
          </motion.div>
        ) : (
          messages.map((message) => (
            <ChatBubble
              key={message.id}
              message={message}
              onPlayAudio={onPlayAudio}
              onStopAudio={onStopAudio}
              speaking={speaking}
              showAudio={ttsSupported}
            />
          ))
        )}

        {isThinking && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="glass-card p-4"
          >
            <div className="flex items-center gap-3">
              <div className="flex gap-1">
                <motion.div
                  className="h-2 w-2 rounded-full bg-primary"
                  animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                  transition={{ repeat: Infinity, duration: 1, delay: 0 }}
                />
                <motion.div
                  className="h-2 w-2 rounded-full bg-primary"
                  animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                  transition={{ repeat: Infinity, duration: 1, delay: 0.2 }}
                />
                <motion.div
                  className="h-2 w-2 rounded-full bg-primary"
                  animate={{ scale: [1, 1.2, 1], opacity: [0.5, 1, 0.5] }}
                  transition={{ repeat: Infinity, duration: 1, delay: 0.4 }}
                />
              </div>
              <span className="text-sm text-white/70">MedIntel is analyzing...</span>
            </div>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Waveform */}
      <div className="px-6">
        <Waveform isActive={listening || speaking || isThinking} />
      </div>

      {/* Composer */}
      <div className="p-6 pt-4">
        <Composer
          onSend={onSendMessage}
          onUpload={onFileUpload}
          onMicToggle={onMicToggle}
          isListening={listening}
          isThinking={isThinking}
          transcript={transcript}
          interimTranscript={interimTranscript}
          onClearTranscript={onClearTranscript}
          sttSupported={sttSupported}
        />
      </div>
    </section>
  );
}

export default ChatCenter;
