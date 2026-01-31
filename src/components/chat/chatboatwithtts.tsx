import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Volume2, VolumeX } from "lucide-react";

export default function ChatbotWithTTS() {
  const [messages, setMessages] = useState<
    { sender: "user" | "bot"; text: string }[]
  >([]);
  const [input, setInput] = useState("");
  const [ttsEnabled, setTtsEnabled] = useState(true);

  const speak = (text: string) => {
    if (!ttsEnabled) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "en-US";
    utterance.rate = 1;
    utterance.pitch = 1;
    window.speechSynthesis.speak(utterance);
  };

  const sendMessage = () => {
    if (!input.trim()) return;

    const userMessage = { sender: "user" as const, text: input };
    setMessages((prev) => [...prev, userMessage]);

    // Mock AI response (replace with API later)
    const botReply =
      "Based on your input, I recommend maintaining hydration and monitoring symptoms.";

    const botMessage = { sender: "bot" as const, text: botReply };

    setTimeout(() => {
      setMessages((prev) => [...prev, botMessage]);
      speak(botReply);
    }, 600);

    setInput("");
  };

  return (
    <div className="max-w-xl mx-auto p-6 bg-card rounded-3xl shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold">AI Health Chatbot</h2>
        <Button
          variant="ghost"
          size="icon"
          onClick={() => setTtsEnabled(!ttsEnabled)}
        >
          {ttsEnabled ? <Volume2 /> : <VolumeX />}
        </Button>
      </div>

      <div className="h-80 overflow-y-auto space-y-3 mb-4">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-3 rounded-xl max-w-[80%] ${
              m.sender === "user"
                ? "ml-auto bg-primary text-primary-foreground"
                : "bg-muted"
            }`}
          >
            {m.text}
          </div>
        ))}
      </div>

      <div className="flex gap-2">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about your health..."
          className="flex-1 border rounded-xl px-4 py-2"
        />
        <Button onClick={sendMessage}>Send</Button>
      </div>
    </div>
  );
}
