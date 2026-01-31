export const SYMPTOMS_LIST = [
  { id: 'headache', label: 'Headache', category: 'neurological' },
  { id: 'fatigue', label: 'Fatigue / Tiredness', category: 'general' },
  { id: 'fever', label: 'Fever', category: 'general' },
  { id: 'cough', label: 'Cough', category: 'respiratory' },
  { id: 'sore_throat', label: 'Sore Throat', category: 'respiratory' },
  { id: 'shortness_breath', label: 'Shortness of Breath', category: 'respiratory' },
  { id: 'chest_pain', label: 'Chest Pain', category: 'cardiovascular' },
  { id: 'palpitations', label: 'Heart Palpitations', category: 'cardiovascular' },
  { id: 'nausea', label: 'Nausea', category: 'gastrointestinal' },
  { id: 'stomach_pain', label: 'Stomach Pain', category: 'gastrointestinal' },
  { id: 'diarrhea', label: 'Diarrhea', category: 'gastrointestinal' },
  { id: 'muscle_pain', label: 'Muscle Pain', category: 'musculoskeletal' },
  { id: 'joint_pain', label: 'Joint Pain', category: 'musculoskeletal' },
  { id: 'dizziness', label: 'Dizziness', category: 'neurological' },
  { id: 'insomnia', label: 'Insomnia / Sleep Issues', category: 'neurological' },
  { id: 'anxiety', label: 'Anxiety', category: 'mental' },
  { id: 'depression', label: 'Low Mood / Depression', category: 'mental' },
  { id: 'stress', label: 'High Stress', category: 'mental' },
  { id: 'skin_rash', label: 'Skin Rash', category: 'dermatological' },
  { id: 'allergies', label: 'Allergies', category: 'immunological' },
  { id: 'weight_change', label: 'Unexplained Weight Change', category: 'metabolic' },
  { id: 'appetite_loss', label: 'Loss of Appetite', category: 'gastrointestinal' },
] as const;

export const SYMPTOM_CATEGORIES = {
  general: { label: 'General', color: 'hsl(var(--chart-1))' },
  respiratory: { label: 'Respiratory', color: 'hsl(var(--chart-2))' },
  cardiovascular: { label: 'Cardiovascular', color: 'hsl(var(--chart-3))' },
  gastrointestinal: { label: 'Digestive', color: 'hsl(var(--chart-4))' },
  neurological: { label: 'Neurological', color: 'hsl(var(--chart-5))' },
  musculoskeletal: { label: 'Musculoskeletal', color: 'hsl(var(--warning))' },
  mental: { label: 'Mental Health', color: 'hsl(var(--info))' },
  dermatological: { label: 'Skin', color: 'hsl(var(--accent))' },
  immunological: { label: 'Immune', color: 'hsl(var(--success))' },
  metabolic: { label: 'Metabolic', color: 'hsl(var(--destructive))' },
} as const;

export const MEDICAL_DISCLAIMER = `
⚕️ Medical Disclaimer: This tool provides general wellness information only and is NOT a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay in seeking it because of something you have read or received from this application.

If you are experiencing a medical emergency, please call emergency services (911) immediately.
`;

export const REGIONS = ['North Region', 'South Region', 'East Region', 'West Region'] as const;