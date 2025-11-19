/**
 * ChatPage - Main orchestrator for MedIntel chat interface
 * 
 * BACKEND INTEGRATION POINTS:
 * 1. POST /api/v1/chat - Send message and get AI response
 *    Body: { question, context, model_provider, student_mode, mode }
 *    Response: { summary, answer, risk_level, confidence, emotion, next_steps, citations }
 * 
 * 2. POST /api/v1/upload - Upload and parse files (PDF/image/audio)
 *    FormData: { file }
 *    Response: { id, filename, extractedText, labs, riskAssessment }
 * 
 * 3. GET /api/v1/history - Fetch chat history
 * 4. GET /api/v1/report/{id} - Fetch specific report
 * 
 * DEMO FLOW:
 * Step 1: Click "Upload" → select sample PDF from public/assets/
 * Step 2: Type "Explain my hemoglobin levels"
 * Step 3: Toggle to "Student Mode" and ask "How does HbA1c work?"
 */

import { useState, useEffect, useMemo } from 'react';
import axios from 'axios';
import Sidebar from '../components/Sidebar.jsx';
import ChatCenter from '../components/ChatCenter.jsx';
import RightPanel from '../components/RightPanel.jsx';
import useTTS from '../hooks/useTTS.js';
import useSTT from '../hooks/useSTT.js';
import useLocalStore from '../hooks/useLocalStore.js';
import { dummyChats, dummyReports, chatModes } from '../data/dummyData.js';

// Configure axios baseURL for backend
axios.defaults.baseURL = 'http://localhost:8000';

function ChatPage() {
  const [activeMode, setActiveMode] = useState('medical');
  const [messages, setMessages] = useState([]);
  const [currentReport, setCurrentReport] = useState(null);
  const [isThinking, setIsThinking] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [rightPanelCollapsed, setRightPanelCollapsed] = useState(false);

  const { speak, cancel, speaking, supported: ttsSupported } = useTTS();
  const {
    transcript,
    interimTranscript,
    listening,
    startListening,
    stopListening,
    resetTranscript,
    supported: sttSupported,
  } = useSTT();

  const {
    chats,
    activeChat,
    setActiveChat,
    saveChat,
    deleteChat,
    renameChat,
    getUserProfile,
    getSessionId,
  } = useLocalStore();

  // Load demo data on mount
  useEffect(() => {
    // Only show chats that exist in localStorage
    // Don't automatically load dummy data
    if (!activeChat && chats.length > 0) {
      const firstChat = chats[0];
      setActiveChat(firstChat);
      setMessages(firstChat.messages || []);
      setActiveMode(firstChat.mode);
    }
  }, [chats, activeChat, setActiveChat]);

  // Compute latest risk level
  const latestRisk = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      if (messages[i].riskLevel) {
        return messages[i].riskLevel;
      }
    }
    return 'Green';
  }, [messages]);

  // Handle sending messages
  const handleSendMessage = async (text, attachments = []) => {
    if (!text?.trim()) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsThinking(true);

    try {
      // Call backend API
      const response = await axios.post('/api/v1/chat', {
        question: text,
        context: currentReport?.extractedText || '',
        model_provider: activeMode === 'student' ? 'openai' : 'gemini',
        student_mode: activeMode === 'student',
        mode: activeMode,
        session_id: getSessionId(),
        user_profile: getUserProfile(),
      });

      // Handle model response with human_line and JSON extraction
      handleModelResponse(response.data);

    } catch (error) {
      console.error('Chat error:', error);
      const fallback = {
        id: crypto.randomUUID(),
        role: 'assistant',
        summary: 'Demo mode active.',
        content:
          'MedIntel is running in local demo mode. Connect the backend at http://localhost:8000 to receive live clinical summaries and structured follow-ups.',
        emotion: 'neutral',
        riskLevel: 'Green',
        confidence: 'N/A',
        nextSteps: ['Connect backend', 'Configure API keys', 'Re-run query'],
        sources: ['Demo Dataset'],
        timestamp: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, fallback]);
    } finally {
      setIsThinking(false);
    }
  };

  // Handle model response with human_line and JSON extraction
  const handleModelResponse = (resp) => {
    // resp can be either: { raw_text, human_line, json } OR { summary, answer, ... }
    const human = resp.human_line?.trim();
    const json = resp.json || resp; // fallback to full response if no json field

    // 1) Add human_line message (if not duplicate)
    if (human) {
      const lastMsg = messages[messages.length - 1];
      const isDuplicate = lastMsg && lastMsg.role === 'assistant' && lastMsg.content.trim() === human;
      
      if (!isDuplicate) {
        appendMessage({
          id: crypto.randomUUID(),
          role: 'assistant',
          content: human,
          meta: { type: 'human_line' },
          timestamp: new Date().toISOString(),
        });
        
        // Speak the human line, then append structured JSON
        speak(human, {
          onStart: () => {},
          onEnd: () => {
            appendStructuredJson(json);
          },
        });
      } else {
        // Duplicate found -> directly append json
        appendStructuredJson(json);
      }
    } else {
      // No human_line present -> directly append structured json
      appendStructuredJson(json);
    }
  };

  // Append a message to the chat
  const appendMessage = (message) => {
    setMessages((prev) => [...prev, message]);
  };

  // Append structured JSON as formatted messages
  const appendStructuredJson = (json) => {
    if (!json) {
      appendMessage({
        id: crypto.randomUUID(),
        role: 'assistant',
        content: "Sorry, I couldn't format a response. Try again.",
        timestamp: new Date().toISOString(),
      });
      return;
    }

    // Create structured message with all metadata
    const structuredMessage = {
      id: crypto.randomUUID(),
      role: 'assistant',
      summary: json.summary || '',
      content: json.answer || json.content || '',
      emotion: json.emotion || 'neutral',
      riskLevel: json.risk_level || 'Green',
      confidence: json.confidence || 'N/A',
      nextSteps: json.next_steps || [],
      sources: json.citations || [],
      meta: { 
        type: 'structured',
        raw_json: json,
      },
      timestamp: new Date().toISOString(),
    };

    appendMessage(structuredMessage);
  };

  // Simulate streaming token-by-token response
  const simulateStreamingResponse = async (question) => {
    const demoResponse = {
      id: crypto.randomUUID(),
      role: 'assistant',
      summary: 'Analysis complete.',
      content: `Based on your question "${question}", here is a comprehensive medical insight. This is a simulated response demonstrating the streaming capability. In production, this would connect to Groq/Gemini/OpenAI for real-time medical analysis with proper citations and risk scoring.`,
      emotion: 'supportive',
      riskLevel: 'Green',
      confidence: '0.85',
      sources: ['Demo Knowledge Base', 'MedIntel Training Data'],
      nextSteps: [
        'Review provided information',
        'Consult with healthcare provider if needed',
        'Track symptoms in MedIntel journal',
      ],
      timestamp: new Date().toISOString(),
    };

    // Create partial message for streaming effect
    const streamingMessage = {
      ...demoResponse,
      content: '',
    };

    setMessages((prev) => [...prev, streamingMessage]);

    const words = demoResponse.content.split(' ');
    for (let i = 0; i < words.length; i++) {
      await new Promise((resolve) => setTimeout(resolve, 50));
      setMessages((prev) => {
        const updated = [...prev];
        const lastMessage = updated[updated.length - 1];
        lastMessage.content = words.slice(0, i + 1).join(' ');
        return updated;
      });
    }

    // Finalize message
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = demoResponse;
      return updated;
    });
  };

  // Handle file uploads
  const handleFileUpload = async (files) => {
    if (!files?.length) return;
    const file = files[0];

    // Show processing state
    setCurrentReport({
      id: crypto.randomUUID(),
      filename: file.name,
      extractedText: 'Processing upload...',
      labs: [],
      uploadedAt: new Date().toISOString(),
    });

    try {
      // TODO: Replace with actual upload
      // const formData = new FormData();
      // formData.append('file', file);
      // const response = await axios.post('/api/v1/upload', formData, {
      //   headers: { 'Content-Type': 'multipart/form-data' },
      // });
      // setCurrentReport(response.data);

      // Use dummy data for now
      await new Promise((resolve) => setTimeout(resolve, 1500));
      setCurrentReport(dummyReports[0]);
    } catch (error) {
      console.error('Upload error:', error);
      setCurrentReport(dummyReports[0]); // Fallback to dummy
    }
  };

  const handleChatSelect = (chat) => {
    setActiveChat(chat);
    setMessages(chat.messages || []);
    setActiveMode(chat.mode || 'medical');
    cancel(); // Stop any ongoing TTS
  };

  const handleDeleteChat = (chatId) => {
    deleteChat(chatId);
    
    // Clear messages if we deleted the active chat
    if (activeChat?.id === chatId) {
      setMessages([]);
      
      // Select the first remaining chat if available
      const remainingChats = chats.filter(c => c.id !== chatId);
      if (remainingChats.length > 0) {
        handleChatSelect(remainingChats[0]);
      }
    }
  };

  const handleNewChat = () => {
    const newChat = {
      id: crypto.randomUUID(),
      title: 'New conversation',
      lastMessage: '',
      timestamp: 'Just now',
      mode: activeMode,
      messages: [],
    };
    setActiveChat(newChat);
    setMessages([]);
    setCurrentReport(null);
  };

  const handlePlayAudio = (text) => {
    if (!ttsSupported || !text) return;
    speak(text);
  };

  return (
    <div className="flex min-h-screen w-full bg-bg-dark text-white">
      <div className="heartbeat-line" aria-hidden="true"></div>

      {/* Left Sidebar */}
      <Sidebar
        chats={chats}
        modes={chatModes}
        activeMode={activeMode}
        onModeChange={setActiveMode}
        onChatSelect={handleChatSelect}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onRenameChat={renameChat}
        collapsed={sidebarCollapsed}
        onToggleCollapse={() => setSidebarCollapsed(!sidebarCollapsed)}
      />

      {/* Center Chat Area */}
      <ChatCenter
        messages={messages}
        isThinking={isThinking}
        activeMode={activeMode}
        onSendMessage={handleSendMessage}
        onFileUpload={handleFileUpload}
        onPlayAudio={handlePlayAudio}
        onStopAudio={cancel}
        speaking={speaking}
        listening={listening}
        onMicToggle={listening ? stopListening : startListening}
        transcript={transcript}
        interimTranscript={interimTranscript}
        onClearTranscript={resetTranscript}
        sttSupported={sttSupported}
        ttsSupported={ttsSupported}
      />

      {/* Right Report Panel */}
      <RightPanel
        report={currentReport}
        riskLevel={latestRisk}
        collapsed={rightPanelCollapsed}
        onToggleCollapse={() => setRightPanelCollapsed(!rightPanelCollapsed)}
      />
    </div>
  );
}

export default ChatPage;
