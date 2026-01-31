import { useState, useRef, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useAuth } from '@/hooks/useAuth';
import { useLifestyleLogs } from '@/hooks/useLifestyleLogs';
import { useProfile } from '@/hooks/useProfile';
import { useVoiceInput } from '@/hooks/useVoiceInput';
import { useToast } from '@/hooks/use-toast';
import {
  Bot,
  Send,
  X,
  MessageCircle,
  Volume2,
  VolumeX,
  Loader2,
  AlertTriangle,
  Stethoscope,
  Mic,
  MicOff,
} from 'lucide-react';
import { cn } from '@/lib/utils';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

export function HealthChatbot() {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voiceEnabled, setVoiceEnabled] = useState(true);
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const { user } = useAuth();
  const { averages } = useLifestyleLogs();
  const { profile } = useProfile();
  const { toast } = useToast();

  const handleVoiceResult = useCallback((transcript: string) => {
    setInput(transcript);
  }, []);

  const handleVoiceError = useCallback((error: string) => {
    toast({
      title: 'Voice Input Error',
      description: error,
      variant: 'destructive',
    });
  }, [toast]);

  const { 
    isListening, 
    isSupported: voiceSupported, 
    toggleListening,
    transcript: liveTranscript,
  } = useVoiceInput({
    onResult: handleVoiceResult,
    onError: handleVoiceError,
  });

  // Update input with live transcript while listening
  useEffect(() => {
    if (isListening && liveTranscript) {
      setInput(liveTranscript);
    }
  }, [isListening, liveTranscript]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight;
    }
  }, [messages]);

  const speakText = useCallback(async (text: string) => {
    if (!voiceEnabled) return;

    try {
      setIsSpeaking(true);
      
      // Clean text for speech (remove markdown formatting)
      const cleanText = text
        .replace(/\*\*/g, '')
        .replace(/\*/g, '')
        .replace(/#{1,6}\s/g, '')
        .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
        .replace(/⚠️/g, 'Warning: ')
        .substring(0, 500); // Limit for TTS

      const response = await fetch(
        `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/health-tts`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            apikey: import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
            Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
          },
          body: JSON.stringify({ text: cleanText }),
        }
      );

      if (!response.ok) {
        console.error('TTS failed:', response.status);
        setIsSpeaking(false);
        return;
      }

      const audioBlob = await response.blob();
      const audioUrl = URL.createObjectURL(audioBlob);
      
      if (audioRef.current) {
        audioRef.current.pause();
      }
      
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      
      audio.onended = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };
      
      audio.onerror = () => {
        setIsSpeaking(false);
        URL.revokeObjectURL(audioUrl);
      };

      await audio.play();
    } catch (error) {
      console.error('TTS error:', error);
      setIsSpeaking(false);
    }
  }, [voiceEnabled]);

  const streamChat = useCallback(async (userMessages: Message[]) => {
    const healthContext = {
      profile: profile ? {
        age: profile.age,
        gender: profile.gender,
        height: profile.height_cm,
        weight: profile.weight_kg,
      } : null,
      lifestyle: averages ? {
        avgSleep: averages.sleep,
        avgExercise: averages.exercise,
        avgSteps: averages.steps,
        avgDiet: averages.diet,
        avgStress: averages.stress,
      } : null,
    };

    const response = await fetch(
      `${import.meta.env.VITE_SUPABASE_URL}/functions/v1/health-chat`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          apikey: import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY,
          Authorization: `Bearer ${import.meta.env.VITE_SUPABASE_PUBLISHABLE_KEY}`,
        },
        body: JSON.stringify({ 
          messages: userMessages,
          healthContext,
        }),
      }
    );

    if (!response.ok || !response.body) {
      throw new Error('Failed to start chat stream');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let textBuffer = '';
    let assistantContent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      textBuffer += decoder.decode(value, { stream: true });

      let newlineIndex: number;
      while ((newlineIndex = textBuffer.indexOf('\n')) !== -1) {
        let line = textBuffer.slice(0, newlineIndex);
        textBuffer = textBuffer.slice(newlineIndex + 1);

        if (line.endsWith('\r')) line = line.slice(0, -1);
        if (line.startsWith(':') || line.trim() === '') continue;
        if (!line.startsWith('data: ')) continue;

        const jsonStr = line.slice(6).trim();
        if (jsonStr === '[DONE]') break;

        try {
          const parsed = JSON.parse(jsonStr);
          const content = parsed.choices?.[0]?.delta?.content;
          if (content) {
            assistantContent += content;
            setMessages(prev => {
              const last = prev[prev.length - 1];
              if (last?.role === 'assistant') {
                return prev.map((m, i) => 
                  i === prev.length - 1 ? { ...m, content: assistantContent } : m
                );
              }
              return [...prev, { role: 'assistant', content: assistantContent }];
            });
          }
        } catch {
          textBuffer = line + '\n' + textBuffer;
          break;
        }
      }
    }

    return assistantContent;
  }, [averages, profile]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = { role: 'user', content: input.trim() };
    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const allMessages = [...messages, userMessage];
      const response = await streamChat(allMessages);
      
      // Speak the response
      if (response && voiceEnabled) {
        await speakText(response);
      }
    } catch (error) {
      console.error('Chat error:', error);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: 'I apologize, but I encountered an error. Please try again. If you\'re experiencing a medical emergency, please call emergency services immediately.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const stopSpeaking = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current = null;
    }
    setIsSpeaking(false);
  };

  if (!isOpen) {
    return (
      <Button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 z-50 h-14 w-14 rounded-full gradient-health shadow-lg shadow-primary/30 hover:shadow-xl hover:shadow-primary/40 hover:scale-110 transition-all"
      >
        <MessageCircle className="h-6 w-6" />
      </Button>
    );
  }

  return (
    <Card className="fixed bottom-6 right-6 z-50 w-[380px] h-[600px] shadow-2xl border-border/50 flex flex-col overflow-hidden">
      <CardHeader className="gradient-health text-primary-foreground py-4 px-4 flex-shrink-0">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-full bg-white/20 flex items-center justify-center">
              <Stethoscope className="h-5 w-5" />
            </div>
            <div>
              <CardTitle className="text-lg font-semibold">Health Assistant</CardTitle>
              <p className="text-xs text-primary-foreground/80">AI-powered health guidance</p>
            </div>
          </div>
          <div className="flex items-center gap-1">
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-primary-foreground hover:bg-white/20"
              onClick={() => setVoiceEnabled(!voiceEnabled)}
            >
              {voiceEnabled ? <Volume2 className="h-4 w-4" /> : <VolumeX className="h-4 w-4" />}
            </Button>
            <Button
              variant="ghost"
              size="icon"
              className="h-8 w-8 text-primary-foreground hover:bg-white/20"
              onClick={() => {
                stopSpeaking();
                setIsOpen(false);
              }}
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>

      <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
        {messages.length === 0 && (
          <div className="text-center py-8">
            <Bot className="h-12 w-12 mx-auto text-muted-foreground/50 mb-4" />
            <p className="text-muted-foreground text-sm mb-2">
              Hello! I'm your AI health assistant.
            </p>
            <p className="text-muted-foreground text-xs mb-4">
              Ask me about health tips, symptoms, or wellness advice.
            </p>
            <div className="bg-warning/10 border border-warning/30 rounded-lg p-3 text-xs text-warning-foreground">
              <AlertTriangle className="h-4 w-4 inline mr-2" />
              For medical emergencies, call emergency services immediately.
            </div>
          </div>
        )}

        <div className="space-y-4">
          {messages.map((message, index) => (
            <div
              key={index}
              className={cn(
                'flex',
                message.role === 'user' ? 'justify-end' : 'justify-start'
              )}
            >
              <div
                className={cn(
                  'max-w-[85%] rounded-2xl px-4 py-2.5 text-sm',
                  message.role === 'user'
                    ? 'bg-primary text-primary-foreground rounded-br-md'
                    : 'bg-muted rounded-bl-md'
                )}
              >
                <p className="whitespace-pre-wrap">{message.content}</p>
              </div>
            </div>
          ))}

          {isLoading && messages[messages.length - 1]?.role === 'user' && (
            <div className="flex justify-start">
              <div className="bg-muted rounded-2xl rounded-bl-md px-4 py-3">
                <Loader2 className="h-4 w-4 animate-spin" />
              </div>
            </div>
          )}

          {isSpeaking && (
            <div className="flex justify-center">
              <Button
                variant="outline"
                size="sm"
                onClick={stopSpeaking}
                className="gap-2"
              >
                <Volume2 className="h-4 w-4 animate-pulse" />
                Speaking... Click to stop
              </Button>
            </div>
          )}
        </div>
      </ScrollArea>

      <CardContent className="p-4 border-t flex-shrink-0">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex gap-2"
        >
          {voiceSupported && (
            <Button
              type="button"
              variant={isListening ? "destructive" : "outline"}
              size="icon"
              onClick={toggleListening}
              disabled={isLoading}
              className={cn(
                "flex-shrink-0 transition-all",
                isListening && "animate-pulse"
              )}
            >
              {isListening ? (
                <MicOff className="h-4 w-4" />
              ) : (
                <Mic className="h-4 w-4" />
              )}
            </Button>
          )}
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={isListening ? "Listening..." : "Describe your health concern..."}
            disabled={isLoading}
            className={cn("flex-1", isListening && "border-primary")}
          />
          <Button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="gradient-health"
          >
            <Send className="h-4 w-4" />
          </Button>
        </form>
        <p className="text-[10px] text-muted-foreground text-center mt-2">
          {isListening ? "🎤 Speak now..." : "Not a substitute for professional medical advice"}
        </p>
      </CardContent>
    </Card>
  );
}
