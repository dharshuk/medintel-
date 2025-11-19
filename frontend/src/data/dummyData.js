// Placeholder seeds for modes, chat history, and parsed reports (ENHANCED).

export const chatModes = [
  { id: 'medical', label: 'Medical', description: 'Clinical-grade guidance', accent: '#23c68b', icon: '🩺' },
  { id: 'general', label: 'General', description: 'Everyday wellness help', accent: '#28F7CE', icon: '💬' },
  { id: 'mental', label: 'Mental Health', description: 'Calm, safe check-ins', accent: '#C084FC', icon: '🧘' },
  { id: 'student', label: 'Student', description: 'Step-by-step coaching', accent: '#FFB800', icon: '🎓' },
];

export const dummyChats = [
  {
    id: 'chat-001',
    title: 'Chest tightness follow-up',
    lastMessage: 'Monitoring mild shortness of breath after workout. Labs reviewed.',
    timestamp: '2h ago',
    mode: 'medical',
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: "I've been feeling tightness in my chest after exercise. Should I be worried?",
        timestamp: '2024-11-19T10:30:00Z',
      },
      {
        id: 'm2',
        role: 'assistant',
        summary: 'Exercise-induced chest tightness assessment.',
        content:
          'Chest tightness post-exercise can have several causes ranging from benign muscle strain to more serious cardiac concerns. Given your age and activity level, muscular causes are more common. However, I recommend monitoring for: radiating pain, shortness of breath at rest, or dizziness. If any of these occur, seek immediate medical attention.',
        emotion: 'supportive',
        riskLevel: 'Amber',
        confidence: '0.82',
        sources: ['Mayo Clinic: Exercise-induced symptoms', 'AHA Guidelines 2024'],
        nextSteps: [
          'Track symptoms in a journal (frequency, duration, intensity)',
          'Schedule a stress test with your cardiologist',
          'Avoid high-intensity workouts until cleared',
        ],
        timestamp: '2024-11-19T10:31:45Z',
      },
    ],
  },
  {
    id: 'chat-002',
    title: 'Mood tracking journal',
    lastMessage: 'Logged energy dips in afternoon. Discussed sleep hygiene.',
    timestamp: '5h ago',
    mode: 'mental',
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: 'I feel really low energy around 3pm every day. Is this normal?',
        timestamp: '2024-11-19T07:00:00Z',
      },
      {
        id: 'm2',
        role: 'assistant',
        summary: 'Afternoon energy dip analysis.',
        content:
          "Afternoon energy slumps are common and can be related to circadian rhythms, blood sugar fluctuations, or sleep quality. To address this: ensure you're getting 7-9 hours of quality sleep, eat balanced meals with protein and complex carbs, stay hydrated, and consider a brief walk or light stretching around 2pm. If fatigue persists despite lifestyle changes, consider discussing with a healthcare provider to rule out thyroid issues or vitamin deficiencies.",
        emotion: 'calm',
        riskLevel: 'Green',
        confidence: '0.75',
        sources: ['Sleep Foundation', 'NIH: Circadian Biology'],
        nextSteps: [
          'Implement a consistent sleep schedule',
          'Log meals and energy levels for 1 week',
          'Try 10-minute walks after lunch',
        ],
        timestamp: '2024-11-19T07:02:30Z',
      },
    ],
  },
  {
    id: 'chat-003',
    title: 'Biochem flashcards',
    lastMessage: 'Requested ATP pathway summary for exam prep.',
    timestamp: '1d ago',
    mode: 'student',
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: 'Can you explain the ATP synthesis pathway step-by-step?',
        timestamp: '2024-11-18T14:00:00Z',
      },
      {
        id: 'm2',
        role: 'assistant',
        summary: 'Step-by-step ATP synthesis explanation.',
        content:
          '**ATP Synthesis Overview (Oxidative Phosphorylation)**\n\n**Step 1: Glycolysis**\n- Location: Cytoplasm\n- Input: 1 Glucose\n- Output: 2 Pyruvate, 2 ATP (net), 2 NADH\n\n**Step 2: Citric Acid Cycle (Krebs Cycle)**\n- Location: Mitochondrial matrix\n- Input: 2 Acetyl-CoA (from pyruvate)\n- Output: 6 NADH, 2 FADH2, 2 ATP\n\n**Step 3: Electron Transport Chain**\n- Location: Inner mitochondrial membrane\n- NADH and FADH2 donate electrons\n- Proton gradient formed across membrane\n- **Result**: ~28-34 ATP via ATP synthase\n\n**Total**: ~30-32 ATP per glucose molecule',
        emotion: 'educational',
        riskLevel: 'Green',
        confidence: '0.95',
        sources: ['Lehninger Biochemistry 8th Ed', 'Khan Academy: Cellular Respiration'],
        nextSteps: [
          'Draw the pathway diagram',
          'Quiz yourself on enzyme names',
          'Practice calculating ATP yield',
        ],
        timestamp: '2024-11-18T14:03:00Z',
      },
    ],
  },
  {
    id: 'chat-004',
    title: 'Lab results review - HbA1c elevated',
    lastMessage: 'Discussed pre-diabetes management strategies.',
    timestamp: '3d ago',
    mode: 'medical',
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: 'My HbA1c came back at 6.3%. What does this mean?',
        timestamp: '2024-11-16T09:00:00Z',
      },
      {
        id: 'm2',
        role: 'assistant',
        summary: 'HbA1c 6.3% indicates pre-diabetes.',
        content:
          'Your HbA1c of 6.3% falls in the pre-diabetes range (5.7-6.4%). This means your average blood sugar over the past 3 months has been elevated but not yet in the diabetic range (≥6.5%). **Good news**: Pre-diabetes is reversible with lifestyle changes.\n\n**Immediate actions**:\n- Reduce refined carbs and added sugars\n- Aim for 150 min/week moderate exercise\n- Monitor fasting glucose weekly\n- Consider meeting with a dietitian\n\n**Follow-up**: Recheck HbA1c in 3 months.',
        emotion: 'informative',
        riskLevel: 'Amber',
        confidence: '0.88',
        sources: ['ADA Standards of Care 2024', 'CDC: Pre-diabetes'],
        nextSteps: [
          'Schedule nutritionist appointment',
          'Start glucose monitoring log',
          'Increase physical activity gradually',
        ],
        timestamp: '2024-11-16T09:04:00Z',
      },
    ],
  },
  {
    id: 'chat-005',
    title: 'Migraine trigger analysis',
    lastMessage: 'Identified caffeine withdrawal as potential trigger.',
    timestamp: '1w ago',
    mode: 'general',
    messages: [
      {
        id: 'm1',
        role: 'user',
        content: 'I get headaches every weekend. Could it be related to my coffee intake?',
        timestamp: '2024-11-12T16:00:00Z',
      },
      {
        id: 'm2',
        role: 'assistant',
        summary: 'Weekend headaches likely caffeine withdrawal.',
        content:
          "Yes, this is a classic pattern! \"Weekend headaches\" are often caused by caffeine withdrawal. If you drink coffee regularly on weekdays (for work/study) but skip or delay it on weekends, your body experiences withdrawal symptoms including headaches, typically 12-24 hours after your last dose.\n\n**Solutions**:\n1. Maintain consistent caffeine timing (even on weekends)\n2. Gradually reduce overall caffeine intake\n3. Stay well-hydrated\n4. Don't skip breakfast on weekends",
        emotion: 'helpful',
        riskLevel: 'Green',
        confidence: '0.79',
        sources: ['Journal of Headache Pain 2023', 'Cleveland Clinic'],
        nextSteps: [
          'Track headache patterns vs caffeine intake',
          'Try consistent wake times 7 days/week',
          'Consider gradual caffeine tapering',
        ],
        timestamp: '2024-11-12T16:02:30Z',
      },
    ],
  },
];

export const dummyReports = [
  {
    id: 'report-001',
    filename: 'Wellness_Visit_Oct_2024.pdf',
    uploadedAt: '2024-11-19T08:30:00Z',
    type: 'pdf',
    extractedText:
      'PATIENT: John Doe | DOB: 1985-03-15 | DATE: 2024-10-28\n\nCOMPLETE BLOOD COUNT:\nHemoglobin: 14.2 g/dL (13.5-17.5)\nHematocrit: 42.1% (38.8-50.0)\nWBC: 7,200/μL (4,500-11,000)\nPlatelets: 245,000/μL (150,000-400,000)\n\nMETABOLIC PANEL:\nGlucose (fasting): 102 mg/dL (70-100)\nHbA1c: 6.2% (<5.7% normal)\nTotal Cholesterol: 195 mg/dL (<200)\nLDL: 138 mg/dL (<100 optimal)\nHDL: 45 mg/dL (>40)\nTriglycerides: 160 mg/dL (<150)\n\nVITAMINS:\nVitamin D: 28 ng/mL (30-100)\n\nNOTES: Patient reports mild fatigue. Recommend dietary modifications and repeat HbA1c in 3 months.',
    labs: [
      { name: 'Hemoglobin', value: '14.2 g/dL', range: '13.5-17.5 g/dL', flag: 'Normal', category: 'CBC' },
      { name: 'WBC', value: '7,200/μL', range: '4,500-11,000/μL', flag: 'Normal', category: 'CBC' },
      { name: 'Platelets', value: '245,000/μL', range: '150,000-400,000/μL', flag: 'Normal', category: 'CBC' },
      { name: 'Glucose (fasting)', value: '102 mg/dL', range: '70-100 mg/dL', flag: 'High', category: 'Metabolic' },
      { name: 'HbA1c', value: '6.2%', range: '<5.7%', flag: 'High', category: 'Metabolic' },
      { name: 'LDL Cholesterol', value: '138 mg/dL', range: '<100 mg/dL', flag: 'Borderline', category: 'Lipid' },
      { name: 'HDL Cholesterol', value: '45 mg/dL', range: '>40 mg/dL', flag: 'Normal', category: 'Lipid' },
      { name: 'Triglycerides', value: '160 mg/dL', range: '<150 mg/dL', flag: 'High', category: 'Lipid' },
      { name: 'Vitamin D', value: '28 ng/mL', range: '30-100 ng/mL', flag: 'Low', category: 'Vitamins' },
    ],
    riskAssessment: {
      overall: 'Amber',
      reason: 'Elevated HbA1c and lipid panel suggest pre-diabetes and dyslipidemia. Vitamin D deficiency noted.',
      recommendations: [
        'Consult with endocrinologist for pre-diabetes management',
        'Start Vitamin D supplementation (2000 IU daily)',
        'Dietary consultation for lipid management',
        'Recheck labs in 3 months',
      ],
    },
  },
  {
    id: 'report-002',
    filename: 'Cardiology_Followup_Nov.pdf',
    uploadedAt: '2024-11-18T14:20:00Z',
    type: 'pdf',
    extractedText:
      'PATIENT: Jane Smith | DOB: 1978-07-22 | DATE: 2024-11-15\n\nECHOCARDIOGRAM RESULTS:\nEjection Fraction: 58% (normal >55%)\nLeft Ventricular Function: Normal\nValves: No significant regurgitation\n\nSTRESS TEST:\nDuration: 9 minutes (Bruce protocol)\nMax Heart Rate: 162 bpm (91% of predicted)\nBlood Pressure Response: Normal\nECG Changes: None\nSymptoms: None\n\nIMPRESSION: Normal cardiac function. No evidence of ischemia. Continue current medications.',
    labs: [
      { name: 'Ejection Fraction', value: '58%', range: '>55%', flag: 'Normal', category: 'Cardiac' },
      { name: 'Max Heart Rate', value: '162 bpm', range: '91% predicted', flag: 'Normal', category: 'Stress Test' },
    ],
    riskAssessment: {
      overall: 'Green',
      reason: 'Normal cardiac function with no signs of ischemia or structural abnormalities.',
      recommendations: [
        'Continue current medication regimen',
        'Annual cardiology follow-up',
        'Maintain regular exercise routine',
      ],
    },
  },
];

export const quickPrompts = [
  { id: 'p1', text: 'Explain my latest lab results', icon: '📊' },
  { id: 'p2', text: 'What are my risk factors?', icon: '⚠️' },
  { id: 'p3', text: 'Summarize my health trends', icon: '📈' },
  { id: 'p4', text: 'Create a wellness plan', icon: '🎯' },
];

export const tonePresets = [
  { id: 'concise', label: 'Concise', desc: 'Brief answers' },
  { id: 'detailed', label: 'Detailed', desc: 'In-depth explanations' },
  { id: 'simple', label: 'Simple', desc: 'Easy language' },
  { id: 'technical', label: 'Technical', desc: 'Medical terminology' },
];
