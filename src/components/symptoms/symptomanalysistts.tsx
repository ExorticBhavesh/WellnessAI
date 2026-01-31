import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Volume2 } from "lucide-react";

export default function SymptomAnalysisWithTTS() {
  const [symptoms, setSymptoms] = useState("");
  const [result, setResult] = useState("");

  const speak = (text: string) => {
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 0.95;
    window.speechSynthesis.speak(utterance);
  };

  const analyzeSymptoms = () => {
    if (!symptoms.trim()) return;

    // Mock AI analysis (replace with backend later)
    const analysis = `
Based on the symptoms you provided, this could be a mild condition.
Please ensure adequate rest, hydration, and consult a doctor if symptoms persist.
    `;

    setResult(analysis.trim());
    speak(analysis);
  };

  return (
    <div className="max-w-xl mx-auto p-6 bg-card rounded-3xl shadow-lg">
      <h2 className="text-xl font-bold mb-4">Symptom Analysis</h2>

      <textarea
        value={symptoms}
        onChange={(e) => setSymptoms(e.target.value)}
        placeholder="Describe your symptoms..."
        className="w-full h-28 border rounded-xl p-4 mb-4"
      />

      <Button onClick={analyzeSymptoms} className="w-full">
        Analyze Symptoms
      </Button>

      {result && (
        <div className="mt-6 p-4 bg-muted rounded-xl relative">
          <Button
            size="icon"
            variant="ghost"
            className="absolute top-2 right-2"
            onClick={() => speak(result)}
          >
            <Volume2 />
          </Button>

          <h3 className="font-semibold mb-2">AI Recommendation</h3>
          <p className="text-sm text-muted-foreground">{result}</p>
        </div>
      )}
    </div>
  );
}
