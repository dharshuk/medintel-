// Placeholder seeds for modes, chat history, and parsed reports.
export const chatModes = [
  { id: 'medical', label: 'Medical', description: 'Clinical-grade guidance', accent: '#23c68b' },
  { id: 'general', label: 'General', description: 'Everyday wellness help', accent: '#3ef6ff' },
  { id: 'mental', label: 'Mental Health', description: 'Calm, safe check-ins', accent: '#f79bff' },
  { id: 'student', label: 'Student', description: 'Step-by-step coaching', accent: '#f6c945' },
];

export const dummyHistory = [
  {
    id: 'sess-001',
    title: 'Chest tightness follow-up',
    lastMessage: 'Monitoring mild shortness of breath after workout.',
    timestamp: '2h ago',
    mode: 'medical',
  },
  {
    id: 'sess-002',
    title: 'Mood tracking journal',
    lastMessage: 'Logged energy dips in afternoon.',
    timestamp: '5h ago',
    mode: 'mental',
  },
  {
    id: 'sess-003',
    title: 'Biochem flashcards',
    lastMessage: 'Requested ATP pathway summary.',
    timestamp: '1d ago',
    mode: 'student',
  },
];

export const mockMessages = [
  {
    id: 'm1',
    role: 'assistant',
    summary: 'Vitals review complete.',
    content:
      'Your submitted vitals look stable overall. Hydration appears adequate, but keep an eye on the intermittent tachycardia spikes when exercising in heat.',
    emotion: 'supportive',
    riskLevel: 'Green',
  },
  {
    id: 'm2',
    role: 'user',
    content: 'Can you double-check the lab report I uploaded and tell me what matters most?',
  },
  {
    id: 'm3',
    role: 'assistant',
    summary: 'Labs parsed successfully.',
    content:
      'Hemoglobin A1c is trending upward (6.2%). Consider a follow-up with your clinician to discuss early glucose management strategies.',
    emotion: 'neutral',
    riskLevel: 'Amber',
  },
];

export const mockReport = {
  id: 'report-123',
  filename: 'Wellness_Visit_Oct.pdf',
  extractedText:
    'CBC within expected range. HbA1c 6.2%. LDL 138 mg/dL. Provider recommends dietary adjustments and repeat labs in 3 months.',
  labs: [
    { name: 'HbA1c', value: '6.2%', range: '<5.7%', flag: 'High' },
    { name: 'LDL', value: '138 mg/dL', range: '<100 mg/dL', flag: 'Borderline' },
    { name: 'Vitamin D', value: '32 ng/mL', range: '30-100 ng/mL', flag: 'Normal' },
  ],
};
