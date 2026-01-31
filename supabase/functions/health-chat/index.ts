import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

const MEDICAL_DISCLAIMER = `

⚠️ **IMPORTANT MEDICAL DISCLAIMER**: This AI assistant provides general health information only and is NOT a substitute for professional medical advice, diagnosis, or treatment. If you're experiencing severe symptoms, persistent pain, difficulty breathing, chest pain, or any medical emergency, please **seek immediate medical attention** or call emergency services. Always consult with a qualified healthcare provider for personalized medical advice.`;

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { messages, healthContext } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    const systemPrompt = `You are a compassionate and knowledgeable AI health assistant. Your role is to:

1. Listen carefully to health concerns and provide helpful, evidence-based information
2. Offer practical health tips and lifestyle recommendations
3. Help users understand their symptoms in general terms
4. Encourage healthy habits and preventive care
5. ALWAYS recommend consulting a healthcare professional for:
   - Severe or worsening symptoms
   - Persistent pain or discomfort
   - Any emergency situations
   - Medication-related questions
   - Specific diagnoses or treatment plans

User's Health Context:
${healthContext ? JSON.stringify(healthContext, null, 2) : 'No specific health data provided'}

CRITICAL RULES:
- Never diagnose conditions or prescribe treatments
- For ANY concerning symptoms, advise visiting a doctor
- Be empathetic and supportive in your responses
- Provide general wellness tips when appropriate
- Always end responses about symptoms with a reminder to see a healthcare provider if symptoms persist or worsen
- If symptoms sound severe (chest pain, difficulty breathing, severe bleeding, loss of consciousness, etc.), IMMEDIATELY advise calling emergency services`;

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          { role: "system", content: systemPrompt },
          ...messages,
        ],
        stream: true,
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again later." }), {
          status: 429,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      if (response.status === 402) {
        return new Response(JSON.stringify({ error: "Service temporarily unavailable." }), {
          status: 402,
          headers: { ...corsHeaders, "Content-Type": "application/json" },
        });
      }
      const errorText = await response.text();
      console.error("AI gateway error:", response.status, errorText);
      return new Response(JSON.stringify({ error: "AI service error" }), {
        status: 500,
        headers: { ...corsHeaders, "Content-Type": "application/json" },
      });
    }

    return new Response(response.body, {
      headers: { ...corsHeaders, "Content-Type": "text/event-stream" },
    });
  } catch (error) {
    console.error("Health chat error:", error);
    return new Response(JSON.stringify({ error: error instanceof Error ? error.message : "Unknown error" }), {
      status: 500,
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  }
});
