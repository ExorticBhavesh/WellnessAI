import { serve } from "https://deno.land/std@0.168.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

serve(async (req) => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const { symptoms } = await req.json();
    const LOVABLE_API_KEY = Deno.env.get("LOVABLE_API_KEY");
    
    if (!LOVABLE_API_KEY) {
      throw new Error("LOVABLE_API_KEY is not configured");
    }

    const response = await fetch("https://ai.gateway.lovable.dev/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${LOVABLE_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model: "google/gemini-3-flash-preview",
        messages: [
          {
            role: "system",
            content: `You are a health risk assessment AI. Analyze symptoms and provide:
1. Risk level: "low", "medium", or "high"
2. Brief analysis (2-3 sentences)
3. 3-4 actionable recommendations

IMPORTANT: This is NOT medical advice. Always recommend consulting healthcare professionals for serious symptoms.

Respond in JSON format: {"riskLevel": "low|medium|high", "analysis": "string", "recommendations": ["string"]}`
          },
          {
            role: "user",
            content: `Analyze these symptoms: ${symptoms.join(", ")}`
          }
        ],
      }),
    });

    if (!response.ok) {
      if (response.status === 429) {
        return new Response(JSON.stringify({ error: "Rate limit exceeded. Please try again later." }), {
          status: 429, headers: { ...corsHeaders, "Content-Type": "application/json" }
        });
      }
      throw new Error("AI gateway error");
    }

    const data = await response.json();
    const content = data.choices?.[0]?.message?.content || "";
    
    // Parse JSON from response
    const jsonMatch = content.match(/\{[\s\S]*\}/);
    if (!jsonMatch) {
      throw new Error("Invalid AI response format");
    }
    
    const result = JSON.parse(jsonMatch[0]);

    return new Response(JSON.stringify(result), {
      headers: { ...corsHeaders, "Content-Type": "application/json" },
    });
  } catch (error) {
    console.error("Error:", error);
    return new Response(JSON.stringify({ 
      riskLevel: "medium",
      analysis: "Unable to complete analysis. Please consult a healthcare professional for accurate assessment.",
      recommendations: ["Consult with a healthcare provider", "Monitor your symptoms", "Stay hydrated and rest"]
    }), {
      status: 200, headers: { ...corsHeaders, "Content-Type": "application/json" }
    });
  }
});