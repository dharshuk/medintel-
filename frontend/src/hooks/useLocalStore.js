// Simple localStorage wrapper for persisting chat sessions

import { useCallback, useEffect, useState } from 'react';

const STORAGE_KEYS = {
  CHATS: 'medintel_chats',
  ACTIVE_CHAT: 'medintel_active_chat',
  SETTINGS: 'medintel_settings',
  TONE: 'medintel_tone',
  SESSION_ID: 'medintel_session_id',
};

// Generate or get session ID
function getSessionId() {
  let sessionId = localStorage.getItem(STORAGE_KEYS.SESSION_ID);
  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(STORAGE_KEYS.SESSION_ID, sessionId);
  }
  return sessionId;
}

function useLocalStore() {
  const [chats, setChats] = useState([]);
  const [activeChat, setActiveChat] = useState(null);
  const [tonePreference, setTonePreference] = useState('warm_personal');

  // Load chats from localStorage on mount
  useEffect(() => {
    try {
      const storedChats = localStorage.getItem(STORAGE_KEYS.CHATS);
      const storedActive = localStorage.getItem(STORAGE_KEYS.ACTIVE_CHAT);
      const storedTone = localStorage.getItem(STORAGE_KEYS.TONE);

      if (storedChats) {
        setChats(JSON.parse(storedChats));
      }

      if (storedActive) {
        setActiveChat(JSON.parse(storedActive));
      }
      
      if (storedTone) {
        setTonePreference(storedTone);
      } else {
        // Set default tone
        localStorage.setItem(STORAGE_KEYS.TONE, 'warm_personal');
      }
    } catch (error) {
      console.error('Error loading from localStorage:', error);
    }
  }, []);

  // Save chats to localStorage whenever they change
  useEffect(() => {
    if (chats.length > 0) {
      try {
        localStorage.setItem(STORAGE_KEYS.CHATS, JSON.stringify(chats));
      } catch (error) {
        console.error('Error saving chats:', error);
      }
    }
  }, [chats]);

  // Save active chat
  useEffect(() => {
    if (activeChat) {
      try {
        localStorage.setItem(STORAGE_KEYS.ACTIVE_CHAT, JSON.stringify(activeChat));
      } catch (error) {
        console.error('Error saving active chat:', error);
      }
    }
  }, [activeChat]);

  const saveChat = useCallback((chat) => {
    setChats((prev) => {
      const existingIndex = prev.findIndex((c) => c.id === chat.id);
      if (existingIndex >= 0) {
        const updated = [...prev];
        updated[existingIndex] = chat;
        return updated;
      }
      return [chat, ...prev];
    });
  }, []);

  const deleteChat = useCallback((chatId) => {
    setChats((prev) => {
      const updated = prev.filter((c) => c.id !== chatId);
      // Also update localStorage immediately
      try {
        if (updated.length === 0) {
          localStorage.removeItem(STORAGE_KEYS.CHATS);
        } else {
          localStorage.setItem(STORAGE_KEYS.CHATS, JSON.stringify(updated));
        }
      } catch (error) {
        console.error('Error updating localStorage on delete:', error);
      }
      return updated;
    });
    
    // Clear active chat if it was deleted
    if (activeChat?.id === chatId) {
      setActiveChat(null);
      try {
        localStorage.removeItem(STORAGE_KEYS.ACTIVE_CHAT);
      } catch (error) {
        console.error('Error clearing active chat:', error);
      }
    }
  }, [activeChat]);

  const renameChat = useCallback((chatId, newTitle) => {
    setChats((prev) =>
      prev.map((chat) =>
        chat.id === chatId ? { ...chat, title: newTitle } : chat
      )
    );
  }, []);

  const clearAllChats = useCallback(() => {
    setChats([]);
    setActiveChat(null);
    localStorage.removeItem(STORAGE_KEYS.CHATS);
    localStorage.removeItem(STORAGE_KEYS.ACTIVE_CHAT);
  }, []);

  const getSetting = useCallback((key, defaultValue = null) => {
    try {
      const settings = JSON.parse(localStorage.getItem(STORAGE_KEYS.SETTINGS) || '{}');
      return settings[key] !== undefined ? settings[key] : defaultValue;
    } catch {
      return defaultValue;
    }
  }, []);

  const saveSetting = useCallback((key, value) => {
    try {
      const settings = JSON.parse(localStorage.getItem(STORAGE_KEYS.SETTINGS) || '{}');
      settings[key] = value;
      localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
    } catch (error) {
      console.error('Error saving setting:', error);
    }
  }, []);

  const saveTonePreference = useCallback((tone) => {
    try {
      localStorage.setItem(STORAGE_KEYS.TONE, tone);
      setTonePreference(tone);
    } catch (error) {
      console.error('Error saving tone preference:', error);
    }
  }, []);

  const getUserProfile = useCallback(() => {
    return {
      preferred_tone: localStorage.getItem(STORAGE_KEYS.TONE) || 'warm_personal',
      name: '',
      history_summary: '',
    };
  }, []);

  const getSessionIdFromStore = useCallback(() => {
    return getSessionId();
  }, []);

  return {
    chats,
    activeChat,
    setActiveChat,
    saveChat,
    deleteChat,
    renameChat,
    clearAllChats,
    getSetting,
    saveSetting,
    tonePreference,
    saveTonePreference,
    getUserProfile,
    getSessionId: getSessionIdFromStore,
  };
}

export default useLocalStore;
