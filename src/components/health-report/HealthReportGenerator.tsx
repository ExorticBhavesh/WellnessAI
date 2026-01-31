import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { useProfile } from '@/hooks/useProfile';
import { useLifestyleLogs } from '@/hooks/useLifestyleLogs';
import { useAuth } from '@/hooks/useAuth';
import { supabase } from '@/integrations/supabase/client';
import { FileText, Download, Loader2, Share2 } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';
import { format } from 'date-fns';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';

export function HealthReportGenerator() {
  const { user } = useAuth();
  const { profile, bmi } = useProfile();
  const { logs, averages } = useLifestyleLogs();
  const { toast } = useToast();
  const [isGenerating, setIsGenerating] = useState(false);

  const calculateHealthScore = () => {
    if (!averages) return 65;
    let score = 50;
    if (averages.sleep >= 7 && averages.sleep <= 9) score += 15;
    else if (averages.sleep >= 6) score += 10;
    else score += 5;
    if (averages.exercise >= 30) score += 15;
    else if (averages.exercise >= 15) score += 10;
    else score += 3;
    if (averages.steps >= 10000) score += 10;
    else if (averages.steps >= 5000) score += 6;
    else score += 2;
    score += (averages.diet / 10) * 10;
    score += ((10 - averages.stress) / 10) * 10;
    return Math.min(100, Math.max(0, score));
  };

  const generatePDF = async () => {
    setIsGenerating(true);
    
    try {
      // Fetch symptom checks
      const { data: symptomChecks } = await supabase
        .from('symptom_checks')
        .select('*')
        .eq('user_id', user?.id)
        .order('created_at', { ascending: false })
        .limit(10);

      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.getWidth();
      const healthScore = calculateHealthScore();
      const reportDate = format(new Date(), 'MMMM d, yyyy');

      // Header with gradient effect
      doc.setFillColor(23, 162, 152); // Primary teal
      doc.rect(0, 0, pageWidth, 45, 'F');
      
      // Logo area
      doc.setFillColor(255, 255, 255);
      doc.circle(25, 22, 8, 'F');
      doc.setFontSize(14);
      doc.setTextColor(23, 162, 152);
      doc.text('♥', 22, 26);
      
      // Title
      doc.setFontSize(24);
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      doc.text('Personal Health Report', 40, 20);
      
      doc.setFontSize(11);
      doc.setFont('helvetica', 'normal');
      doc.text(`Generated: ${reportDate}`, 40, 30);

      // Patient Information Section
      let yPos = 60;
      
      doc.setFontSize(14);
      doc.setTextColor(50, 50, 50);
      doc.setFont('helvetica', 'bold');
      doc.text('Patient Information', 14, yPos);
      
      doc.setDrawColor(23, 162, 152);
      doc.setLineWidth(0.5);
      doc.line(14, yPos + 3, 80, yPos + 3);
      
      yPos += 15;
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(80, 80, 80);

      const patientInfo = [
        ['Name', profile?.full_name || 'Not provided'],
        ['Email', user?.email || 'Not provided'],
        ['Age', profile?.age ? `${profile.age} years` : 'Not provided'],
        ['Gender', profile?.gender ? profile.gender.charAt(0).toUpperCase() + profile.gender.slice(1) : 'Not provided'],
        ['Height', profile?.height_cm ? `${profile.height_cm} cm` : 'Not provided'],
        ['Weight', profile?.weight_kg ? `${profile.weight_kg} kg` : 'Not provided'],
        ['BMI', bmi || 'Not calculated'],
      ];

      patientInfo.forEach(([label, value]) => {
        doc.setFont('helvetica', 'bold');
        doc.text(`${label}:`, 14, yPos);
        doc.setFont('helvetica', 'normal');
        doc.text(String(value), 50, yPos);
        yPos += 7;
      });

      // Health Score Section
      yPos += 10;
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(50, 50, 50);
      doc.text('Overall Health Score', 14, yPos);
      doc.line(14, yPos + 3, 80, yPos + 3);

      yPos += 15;
      
      // Score circle
      const scoreColor = healthScore >= 70 ? [34, 197, 94] : healthScore >= 50 ? [245, 158, 11] : [239, 68, 68];
      doc.setFillColor(scoreColor[0], scoreColor[1], scoreColor[2]);
      doc.circle(35, yPos + 10, 15, 'F');
      doc.setFontSize(18);
      doc.setTextColor(255, 255, 255);
      doc.setFont('helvetica', 'bold');
      doc.text(healthScore.toFixed(0), 29, yPos + 14);

      doc.setFontSize(10);
      doc.setTextColor(80, 80, 80);
      doc.setFont('helvetica', 'normal');
      const scoreLabel = healthScore >= 70 ? 'Good' : healthScore >= 50 ? 'Moderate' : 'Needs Improvement';
      doc.text(`Status: ${scoreLabel}`, 60, yPos + 8);
      doc.text('Based on lifestyle data from the past 7 days', 60, yPos + 15);

      // Lifestyle Averages Section
      yPos += 40;
      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(50, 50, 50);
      doc.text('7-Day Lifestyle Averages', 14, yPos);
      doc.line(14, yPos + 3, 100, yPos + 3);

      yPos += 10;
      
      const lifestyleData = [
        ['Metric', 'Average', 'Recommended', 'Status'],
        ['Sleep', `${averages.sleep.toFixed(1)} hours`, '7-9 hours', averages.sleep >= 7 ? '✓ Good' : '⚠ Low'],
        ['Exercise', `${averages.exercise.toFixed(0)} min`, '30+ min/day', averages.exercise >= 30 ? '✓ Good' : '⚠ Low'],
        ['Daily Steps', averages.steps.toLocaleString(), '10,000+', averages.steps >= 10000 ? '✓ Good' : '⚠ Low'],
        ['Diet Quality', `${averages.diet.toFixed(1)}/10`, '7+/10', averages.diet >= 7 ? '✓ Good' : '⚠ Improve'],
        ['Stress Level', `${averages.stress.toFixed(1)}/10`, '< 5/10', averages.stress <= 5 ? '✓ Good' : '⚠ High'],
      ];

      autoTable(doc, {
        startY: yPos,
        head: [lifestyleData[0]],
        body: lifestyleData.slice(1),
        theme: 'striped',
        headStyles: { fillColor: [23, 162, 152], textColor: 255, fontStyle: 'bold' },
        alternateRowStyles: { fillColor: [245, 250, 249] },
        styles: { fontSize: 9, cellPadding: 4 },
      });

      // Recent Symptom Checks
      if (symptomChecks && symptomChecks.length > 0) {
        yPos = (doc as any).lastAutoTable.finalY + 20;
        
        if (yPos > 240) {
          doc.addPage();
          yPos = 20;
        }

        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(50, 50, 50);
        doc.text('Recent Symptom Assessments', 14, yPos);
        doc.line(14, yPos + 3, 110, yPos + 3);

        yPos += 10;

        const symptomData = symptomChecks.slice(0, 5).map(check => [
          format(new Date(check.created_at), 'MMM d, yyyy'),
          check.symptoms.slice(0, 3).join(', ') + (check.symptoms.length > 3 ? '...' : ''),
          check.risk_level?.toUpperCase() || 'N/A',
        ]);

        autoTable(doc, {
          startY: yPos,
          head: [['Date', 'Symptoms', 'Risk Level']],
          body: symptomData,
          theme: 'striped',
          headStyles: { fillColor: [23, 162, 152], textColor: 255, fontStyle: 'bold' },
          alternateRowStyles: { fillColor: [245, 250, 249] },
          styles: { fontSize: 9, cellPadding: 4 },
          columnStyles: { 
            1: { cellWidth: 80 },
            2: { halign: 'center' }
          },
        });
      }

      // Lifestyle Log History
      if (logs && logs.length > 0) {
        yPos = (doc as any).lastAutoTable.finalY + 20;
        
        if (yPos > 200) {
          doc.addPage();
          yPos = 20;
        }

        doc.setFontSize(14);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(50, 50, 50);
        doc.text('Recent Lifestyle Logs', 14, yPos);
        doc.line(14, yPos + 3, 85, yPos + 3);

        yPos += 10;

        const logData = logs.slice(0, 7).map(log => [
          format(new Date(log.log_date), 'MMM d'),
          `${log.sleep_hours}h`,
          `${log.exercise_minutes} min`,
          log.daily_steps?.toLocaleString() || '-',
          `${log.diet_quality}/10`,
          `${log.stress_level}/10`,
        ]);

        autoTable(doc, {
          startY: yPos,
          head: [['Date', 'Sleep', 'Exercise', 'Steps', 'Diet', 'Stress']],
          body: logData,
          theme: 'striped',
          headStyles: { fillColor: [23, 162, 152], textColor: 255, fontStyle: 'bold' },
          alternateRowStyles: { fillColor: [245, 250, 249] },
          styles: { fontSize: 8, cellPadding: 3 },
        });
      }

      // Recommendations Section
      yPos = (doc as any).lastAutoTable.finalY + 20;
      
      if (yPos > 230) {
        doc.addPage();
        yPos = 20;
      }

      doc.setFontSize(14);
      doc.setFont('helvetica', 'bold');
      doc.setTextColor(50, 50, 50);
      doc.text('AI-Generated Recommendations', 14, yPos);
      doc.line(14, yPos + 3, 115, yPos + 3);

      yPos += 12;
      doc.setFontSize(10);
      doc.setFont('helvetica', 'normal');
      doc.setTextColor(80, 80, 80);

      const recommendations = [];
      if (averages.sleep < 7) recommendations.push('• Aim for 7-9 hours of sleep per night to improve recovery and cognitive function');
      if (averages.exercise < 30) recommendations.push('• Increase daily physical activity to at least 30 minutes for cardiovascular health');
      if (averages.steps < 10000) recommendations.push('• Try to reach 10,000 steps daily - consider walking meetings or evening walks');
      if (averages.diet < 7) recommendations.push('• Focus on whole foods, vegetables, and lean proteins to improve diet quality');
      if (averages.stress > 5) recommendations.push('• Practice stress management techniques like meditation, deep breathing, or yoga');
      if (recommendations.length === 0) recommendations.push('• Great job! Maintain your current healthy lifestyle habits');

      recommendations.forEach(rec => {
        const lines = doc.splitTextToSize(rec, 180);
        doc.text(lines, 14, yPos);
        yPos += lines.length * 6 + 4;
      });

      // Footer / Disclaimer
      const pageCount = doc.getNumberOfPages();
      for (let i = 1; i <= pageCount; i++) {
        doc.setPage(i);
        const footerY = doc.internal.pageSize.getHeight() - 15;
        
        doc.setFillColor(248, 250, 252);
        doc.rect(0, footerY - 5, pageWidth, 25, 'F');
        
        doc.setFontSize(8);
        doc.setTextColor(120, 120, 120);
        doc.setFont('helvetica', 'italic');
        doc.text(
          'DISCLAIMER: This report is for informational purposes only and does not constitute medical advice.',
          14, footerY
        );
        doc.text(
          'Please consult a healthcare professional for diagnosis and treatment.',
          14, footerY + 5
        );
        doc.text(`Page ${i} of ${pageCount}`, pageWidth - 30, footerY);
        doc.text('Generated by WellnessAI', pageWidth / 2 - 20, footerY + 5);
      }

      // Save the PDF
      const fileName = `health-report-${format(new Date(), 'yyyy-MM-dd')}.pdf`;
      doc.save(fileName);

      toast({
        title: 'Report Generated!',
        description: 'Your health report has been downloaded.',
      });
    } catch (error) {
      console.error('Error generating PDF:', error);
      toast({
        title: 'Error',
        description: 'Failed to generate report. Please try again.',
        variant: 'destructive',
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Button 
      onClick={generatePDF} 
      disabled={isGenerating}
      variant="outline"
      className="gap-2 border-primary/30 hover:bg-primary/5 hover:border-primary/50"
    >
      {isGenerating ? (
        <>
          <Loader2 className="h-4 w-4 animate-spin" />
          Generating...
        </>
      ) : (
        <>
          <FileText className="h-4 w-4" />
          Download Report
        </>
      )}
    </Button>
  );
}
