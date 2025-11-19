// Main orchestrator that wires stateful chat, uploads, and voice UX.
import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import ChatSidebar from '../components/ChatSidebar.jsx';
import ChatBubble from '../components/ChatBubble.jsx';
import ChatComposer from '../components/ChatComposer.jsx';
import Avatar from '../components/Avatar.jsx';
import Waveform from '../components/Waveform.jsx';
import ReportPanel from '../components/ReportPanel.jsx';
import useTTS from '../hooks/useTTS.js';
import useSTT from '../hooks/useSTT.js';
import { chatModes, dummyHistory, mockMessages, mockReport } from '../data/dummyHistory.js';

const MODE_TO_MODEL = {
  medical: 'groq',
  general: 'groq',
  mental: 'groq',
  student: 'openai',
};

function Chat() {
  const [messages, setMessages] = useState(mockMessages);
  const [activeMode, setActiveMode] = useState('medical');
  const [currentReport, setCurrentReport] = useState(mockReport);
  const [isThinking, setIsThinking] = useState(false);
  const [selectedHistory, setSelectedHistory] = useState(dummyHistory[0]);
  const [waveValues, setWaveValues] = useState(() => Array.from({ length: 24 }, () => Math.random()));
  const { speak, cancel, speaking, supported: ttsSupported } = useTTS();
  const {
    transcript,
    listening,
    startListening,
    stopListening,
    resetTranscript,
    supported: sttSupported,
  } = useSTT();

  useEffect(() => {
    const interval = setInterval(() => {
      setWaveValues((prev) => prev.map(() => Math.random()));
    }, listening || isThinking ? 120 : 600);
    return () => clearInterval(interval);
  }, [listening, isThinking]);

  const contextualReportText = useMemo(() => currentReport?.extractedText ?? '', [currentReport]);
  const latestRisk = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].riskLevel) {
        return messages[i].riskLevel;
      }
    }
    return 'Green';
  }, [messages]);

  const sendToBackend = async (content, attachments = []) => {
    const providerOverride = MODE_TO_MODEL[activeMode] ?? 'groq';
    const payload = {
      question: content,
      context: contextualReportText,
      model_provider: providerOverride,
      student_mode: activeMode === 'student',
      attachments,
      mode: activeMode,
    };

    const { data } = await axios.post('/api/chat', payload);
    return data;
  };

  const handleSendMessage = async (inputText, attachments = []) => {
    const userText = inputText?.trim();
    if (!userText) return;

    setMessages((prev) => [
      ...prev,
      {
        id: crypto.randomUUID(),
        role: 'user',
        content: userText,
        createdAt: new Date().toISOString(),
      },
    ]);

    setIsThinking(true);
    try {
      const response = await sendToBackend(userText, attachments);
      const assistantMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        summary: response.summary,
        content: response.answer,
        emotion: response.emotion,
        riskLevel: response.risk_level,
        confidence: response.confidence,
        nextSteps: response.next_steps,
        citations: response.citations ?? ['Awaiting sources'],
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error('chat error', error);
      const fallback = {
        id: crypto.randomUUID(),
        role: 'assistant',
        summary: 'Demo mode placeholder.',
        content:
          'MedIntel is running in local demo mode. Connect the backend to receive live clinical summaries and structured follow-ups.',
        emotion: 'neutral',
        riskLevel: 'Green',
        confidence: 'N/A',
        nextSteps: ['Connect backend', 'Re-run query'],
        citations: ['Demo Dataset'],
      };
      setMessages((prev) => [...prev, fallback]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleFileUpload = async (fileList) => {
    if (!fileList?.length) return;
    const file = fileList[0];
    setCurrentReport({
      id: crypto.randomUUID(),
      filename: file.name,
      extractedText: 'Uploading to MedIntel backend…',
      labs: [],
    });

    try {
      const data = new FormData();
      data.append('file', file);
      const response = await axios.post('/api/upload', data, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setCurrentReport(response.data ?? mockReport);
    } catch (error) {
      console.error('upload error', error);
      setCurrentReport(mockReport);
    }
  };

  const handleModeChange = (modeId) => {
    setActiveMode(modeId);
    setMessages([]);
    cancel();
  };

  const handlePlayAudio = (text) => {
    if (!ttsSupported || !text) return;
    speak(text);
  };

  const handleMicToggle = () => {
    if (!sttSupported) return;
    if (listening) {
      stopListening();
      return;
    }
    startListening();
  };

  return (
    <div className="relative flex min-h-screen w-full gap-4 bg-midnight p-4 text-white">
      <div className="heartbeat-line" aria-hidden="true"></div>
      <aside className="w-64 flex-shrink-0">
        <ChatSidebar
          modes={chatModes}
          history={dummyHistory}
          activeMode={activeMode}
          onModeChange={handleModeChange}
          selectedHistory={selectedHistory}
          onSelectHistory={setSelectedHistory}
        />
      </aside>

      <section className="medintel-bg relative flex flex-1 flex-col overflow-hidden rounded-3xl border border-white/5 px-6 py-4 shadow-glow">
        <Avatar
          mode={activeMode}
          isThinking={isThinking}
          isSpeaking={speaking}
          isListening={listening}
        />
        <div className="mt-4 flex-1 space-y-4 overflow-y-auto pr-2">
          {messages.map((message) => (
            <ChatBubble
              key={message.id}
              message={message}
              onPlayAudio={handlePlayAudio}
              showAudio={ttsSupported}
            />
          ))}
          {isThinking && (
            <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/70">
              MedIntel is synthesizing a safe response…
            </div>
          )}
        </div>
        <Waveform values={waveValues} isActive={listening || speaking || isThinking} />
        <ChatComposer
          onSend={handleSendMessage}
          onUpload={handleFileUpload}
          onMicToggle={handleMicToggle}
          isListening={listening}
          isThinking={isThinking}
          transcript={transcript}
          onClearTranscript={resetTranscript}
          sttSupported={sttSupported}
        />
      </section>

      <aside className="w-80 flex-shrink-0">
        <ReportPanel report={currentReport} riskLevel={latestRisk} />
      </aside>
    </div>
  );
}

export default Chat;
