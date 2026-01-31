import { serve } from "https://deno.land/std@0.190.0/http/server.ts";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
};

interface SummaryEmailRequest {
  userId: string;
  email: string;
  summaryType: 'weekly' | 'monthly';
  summaryData: {
    healthScore: number;
    sleepAvg: number;
    exerciseAvg: number;
    stepsAvg: number;
    dietAvg: number;
    stressAvg: number;
    periodStart: string;
    periodEnd: string;
    recommendations: string[];
  };
}

const handler = async (req: Request): Promise<Response> => {
  if (req.method === "OPTIONS") {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const resendApiKey = Deno.env.get("RESEND_API_KEY");
    
    if (!resendApiKey) {
      console.log("RESEND_API_KEY not configured - email not sent");
      return new Response(
        JSON.stringify({ 
          success: false, 
          message: "Email service not configured. Please add RESEND_API_KEY to enable email notifications." 
        }),
        { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
      );
    }

    const { email, summaryType, summaryData }: SummaryEmailRequest = await req.json();

    if (!email || !summaryType || !summaryData) {
      throw new Error("Missing required fields");
    }

    const isWeekly = summaryType === 'weekly';
    const periodLabel = isWeekly ? 'Weekly' : 'Monthly';
    const healthStatus = summaryData.healthScore >= 70 ? 'Excellent' : summaryData.healthScore >= 50 ? 'Good' : 'Needs Attention';
    const healthColor = summaryData.healthScore >= 70 ? '#22c55e' : summaryData.healthScore >= 50 ? '#f59e0b' : '#ef4444';

    const emailHtml = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${periodLabel} Health Summary</title>
</head>
<body style="margin: 0; padding: 0; background-color: #f8fafc; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background-color: #f8fafc; padding: 40px 20px;">
    <tr>
      <td align="center">
        <table width="600" cellpadding="0" cellspacing="0" style="background-color: #ffffff; border-radius: 16px; overflow: hidden; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);">
          <!-- Header -->
          <tr>
            <td style="background: linear-gradient(135deg, #14b8a6 0%, #22d3ee 100%); padding: 32px; text-align: center;">
              <h1 style="color: #ffffff; margin: 0; font-size: 28px; font-weight: 700;">WellnessAI</h1>
              <p style="color: rgba(255,255,255,0.9); margin: 8px 0 0 0; font-size: 16px;">Your ${periodLabel} Health Summary</p>
            </td>
          </tr>
          
          <!-- Health Score -->
          <tr>
            <td style="padding: 32px; text-align: center;">
              <p style="color: #64748b; margin: 0 0 16px 0; font-size: 14px;">${summaryData.periodStart} - ${summaryData.periodEnd}</p>
              <div style="width: 120px; height: 120px; border-radius: 50%; background: ${healthColor}; margin: 0 auto 16px; display: flex; align-items: center; justify-content: center;">
                <span style="color: #ffffff; font-size: 36px; font-weight: 700; line-height: 120px;">${Math.round(summaryData.healthScore)}</span>
              </div>
              <h2 style="color: #1e293b; margin: 0; font-size: 24px;">Health Score: ${healthStatus}</h2>
            </td>
          </tr>
          
          <!-- Stats Grid -->
          <tr>
            <td style="padding: 0 32px 32px;">
              <table width="100%" cellpadding="0" cellspacing="0">
                <tr>
                  <td width="50%" style="padding: 12px;">
                    <div style="background: #f1f5f9; border-radius: 12px; padding: 16px; text-align: center;">
                      <p style="color: #64748b; margin: 0 0 4px 0; font-size: 12px;">Sleep Average</p>
                      <p style="color: #1e293b; margin: 0; font-size: 24px; font-weight: 700;">${summaryData.sleepAvg.toFixed(1)}h</p>
                    </div>
                  </td>
                  <td width="50%" style="padding: 12px;">
                    <div style="background: #f1f5f9; border-radius: 12px; padding: 16px; text-align: center;">
                      <p style="color: #64748b; margin: 0 0 4px 0; font-size: 12px;">Exercise/Day</p>
                      <p style="color: #1e293b; margin: 0; font-size: 24px; font-weight: 700;">${Math.round(summaryData.exerciseAvg)} min</p>
                    </div>
                  </td>
                </tr>
                <tr>
                  <td width="50%" style="padding: 12px;">
                    <div style="background: #f1f5f9; border-radius: 12px; padding: 16px; text-align: center;">
                      <p style="color: #64748b; margin: 0 0 4px 0; font-size: 12px;">Steps Average</p>
                      <p style="color: #1e293b; margin: 0; font-size: 24px; font-weight: 700;">${summaryData.stepsAvg.toLocaleString()}</p>
                    </div>
                  </td>
                  <td width="50%" style="padding: 12px;">
                    <div style="background: #f1f5f9; border-radius: 12px; padding: 16px; text-align: center;">
                      <p style="color: #64748b; margin: 0 0 4px 0; font-size: 12px;">Diet Quality</p>
                      <p style="color: #1e293b; margin: 0; font-size: 24px; font-weight: 700;">${summaryData.dietAvg.toFixed(1)}/10</p>
                    </div>
                  </td>
                </tr>
              </table>
            </td>
          </tr>
          
          <!-- Recommendations -->
          ${summaryData.recommendations.length > 0 ? `
          <tr>
            <td style="padding: 0 32px 32px;">
              <h3 style="color: #1e293b; margin: 0 0 16px 0; font-size: 18px;">AI Recommendations</h3>
              <ul style="color: #64748b; margin: 0; padding-left: 20px; line-height: 1.8;">
                ${summaryData.recommendations.map(rec => `<li>${rec}</li>`).join('')}
              </ul>
            </td>
          </tr>
          ` : ''}
          
          <!-- CTA -->
          <tr>
            <td style="padding: 0 32px 32px; text-align: center;">
              <a href="${Deno.env.get('SUPABASE_URL')?.replace('.supabase.co', '.lovable.app')}/dashboard" style="display: inline-block; background: linear-gradient(135deg, #14b8a6 0%, #22d3ee 100%); color: #ffffff; text-decoration: none; padding: 14px 32px; border-radius: 12px; font-weight: 600; font-size: 16px;">View Full Dashboard</a>
            </td>
          </tr>
          
          <!-- Footer -->
          <tr>
            <td style="background: #f8fafc; padding: 24px 32px; text-align: center; border-top: 1px solid #e2e8f0;">
              <p style="color: #94a3b8; margin: 0; font-size: 12px;">This email was sent by WellnessAI. You can manage your notification preferences in your account settings.</p>
              <p style="color: #94a3b8; margin: 8px 0 0 0; font-size: 12px;">© 2024 WellnessAI. For informational purposes only.</p>
            </td>
          </tr>
        </table>
      </td>
    </tr>
  </table>
</body>
</html>
    `;

    const emailResponse = await fetch("https://api.resend.com/emails", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${resendApiKey}`,
      },
      body: JSON.stringify({
        from: "WellnessAI <noreply@wellnessai.app>",
        to: [email],
        subject: `Your ${periodLabel} Health Summary - Score: ${Math.round(summaryData.healthScore)}`,
        html: emailHtml,
      }),
    });

    const result = await emailResponse.json();

    if (!emailResponse.ok) {
      throw new Error(result.message || "Failed to send email");
    }

    console.log("Health summary email sent successfully:", result);

    return new Response(
      JSON.stringify({ success: true, id: result.id }),
      { status: 200, headers: { "Content-Type": "application/json", ...corsHeaders } }
    );
  } catch (error: any) {
    console.error("Error in send-health-summary function:", error);
    return new Response(
      JSON.stringify({ error: error.message }),
      { status: 500, headers: { "Content-Type": "application/json", ...corsHeaders } }
    );
  }
};

serve(handler);
