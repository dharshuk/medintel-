// Enhanced sidebar with chat history, mode selector, and search

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

function Sidebar({
  chats,
  modes,
  activeMode,
  onModeChange,
  onChatSelect,
  onNewChat,
  onDeleteChat,
  onRenameChat,
  collapsed,
  onToggleCollapse,
}) {
  const [searchQuery, setSearchQuery] = useState('');
  const [editingChatId, setEditingChatId] = useState(null);
  const [editTitle, setEditTitle] = useState('');

  const filteredChats = chats.filter((chat) =>
    chat.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const handleRename = (chatId) => {
    if (editTitle.trim()) {
      onRenameChat(chatId, editTitle);
    }
    setEditingChatId(null);
    setEditTitle('');
  };

  if (collapsed) {
    return (
      <div className="glass-card flex w-16 flex-col items-center gap-4 p-4">
        <button
          onClick={onToggleCollapse}
          className="rounded-xl p-2 text-white/60 hover:bg-white/10 hover:text-white"
          aria-label="Expand sidebar"
        >
          ☰
        </button>
      </div>
    );
  }

  return (
    <aside className="glass-card flex w-80 flex-shrink-0 flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold text-primary">MedIntel</h2>
        <button
          onClick={onToggleCollapse}
          className="rounded-xl p-2 text-white/60 hover:bg-white/10 hover:text-white"
          aria-label="Collapse sidebar"
        >
          ←
        </button>
      </div>

      {/* New Chat Button */}
      <button onClick={onNewChat} className="btn-primary w-full">
        + New Chat
      </button>

      {/* Mode Selector */}
      <div>
        <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">
          Mode
        </p>
        <div className="grid grid-cols-2 gap-2">
          {modes.map((mode) => {
            const isActive = activeMode === mode.id;
            return (
              <motion.button
                key={mode.id}
                layout
                onClick={() => onModeChange(mode.id)}
                className={`glass-card-hover rounded-xl p-3 text-left transition-all ${
                  isActive ? 'border-primary/60 bg-primary/10 shadow-glow' : ''
                }`}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="mb-1 text-2xl">{mode.icon}</div>
                <p className="text-sm font-semibold" style={{ color: isActive ? mode.accent : '#fff' }}>
                  {mode.label}
                </p>
                <p className="text-xs text-white/60">{mode.description}</p>
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* Search */}
      <input
        type="text"
        placeholder="Search chats..."
        value={searchQuery}
        onChange={(e) => setSearchQuery(e.target.value)}
        className="input-field"
      />

      {/* Chat History */}
      <div className="flex-1 space-y-2 overflow-y-auto">
        <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">
          Recent Chats
        </p>
        <AnimatePresence>
          {filteredChats.map((chat) => (
            <motion.div
              key={chat.id}
              layout
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, x: -20 }}
              className="glass-card-hover group relative cursor-pointer p-3"
              onClick={() => onChatSelect(chat)}
            >
              {editingChatId === chat.id ? (
                <input
                  type="text"
                  value={editTitle}
                  onChange={(e) => setEditTitle(e.target.value)}
                  onBlur={() => handleRename(chat.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') handleRename(chat.id);
                    if (e.key === 'Escape') setEditingChatId(null);
                  }}
                  className="input-field w-full p-1 text-sm"
                  autoFocus
                  onClick={(e) => e.stopPropagation()}
                />
              ) : (
                <>
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <p className="text-sm font-semibold text-white">{chat.title}</p>
                      <p className="text-xs text-white/60">{chat.lastMessage}</p>
                    </div>
                    <div className="flex gap-1 opacity-0 transition-opacity group-hover:opacity-100">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          setEditingChatId(chat.id);
                          setEditTitle(chat.title);
                        }}
                        className="rounded p-1 text-xs text-white/60 hover:bg-white/10 hover:text-white"
                        aria-label="Rename chat"
                      >
                        ✏️
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onDeleteChat(chat.id);
                        }}
                        className="rounded p-1 text-xs text-white/60 hover:bg-white/10 hover:text-danger"
                        aria-label="Delete chat"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>
                  <p className="mt-1 text-[10px] uppercase tracking-widest text-white/40">
                    {chat.timestamp}
                  </p>
                </>
              )}
            </motion.div>
          ))}
        </AnimatePresence>
      </div>

      {/* Footer Info */}
      <div className="rounded-xl border border-primary/30 bg-primary/5 p-3 text-center text-xs text-white/80">
        🔒 HIPAA-ready logging
      </div>
    </aside>
  );
}

export default Sidebar;
