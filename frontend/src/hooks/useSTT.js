// Enhanced Web Speech recognition hook with continuous listening and error recovery

import { useCallback, useEffect, useRef, useState } from 'react';

function useSTT() {
  const [listening, setListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [interimTranscript, setInterimTranscript] = useState('');
  const [error, setError] = useState(null);
  const recognitionRef = useRef(null);
  const supported =
    typeof window !== 'undefined' && (window.SpeechRecognition || window.webkitSpeechRecognition);

  const resetTranscript = useCallback(() => {
    setTranscript('');
    setInterimTranscript('');
  }, []);

  useEffect(() => {
    if (!supported) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognizer = new SpeechRecognition();
    recognizer.lang = 'en-US';
    recognizer.continuous = false;
    recognizer.interimResults = true;
    recognizer.maxAlternatives = 1;

    recognizer.onresult = (event) => {
      let interim = '';
      let final = '';

      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcriptPiece = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          final += transcriptPiece;
        } else {
          interim += transcriptPiece;
        }
      }

      setInterimTranscript(interim);
      if (final) {
        setTranscript((prev) => (prev ? `${prev} ${final}` : final).trim());
      }
    };

    recognizer.onend = () => {
      setListening(false);
      setInterimTranscript('');
    };

    recognizer.onerror = (event) => {
      console.error('Speech recognition error:', event.error);
      setError(event.error);
      setListening(false);
      setInterimTranscript('');
    };

    recognizer.onstart = () => {
      setError(null);
    };

    recognitionRef.current = recognizer;
  }, [supported]);

  const startListening = useCallback(() => {
    if (!supported || !recognitionRef.current) {
      setError('Speech recognition not supported');
      return;
    }

    try {
      recognitionRef.current.start();
      setListening(true);
      setError(null);
    } catch (err) {
      console.error('Error starting recognition:', err);
      setError(err.message);
    }
  }, [supported]);

  const stopListening = useCallback(() => {
    if (!supported || !recognitionRef.current) return;
    try {
      recognitionRef.current.stop();
      setListening(false);
    } catch (err) {
      console.error('Error stopping recognition:', err);
    }
  }, [supported]);

  return {
    transcript,
    interimTranscript,
    listening,
    error,
    supported: Boolean(supported),
    startListening,
    stopListening,
    resetTranscript,
  };
}

export default useSTT;
