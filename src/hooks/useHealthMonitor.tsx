import { useMemo } from 'react';
import { useLifestyleLogs } from './useLifestyleLogs';

interface HealthTip {
  id: string;
  category: 'sleep' | 'exercise' | 'diet' | 'stress' | 'hydration' | 'general';
  priority: 'high' | 'medium' | 'low';
  title: string;
  description: string;
  icon: string;
}

interface HealthAlert {
  id: string;
  severity: 'warning' | 'info' | 'success';
  message: string;
  action?: string;
}

export function useHealthMonitor() {
  const { logs, averages, isLoading } = useLifestyleLogs();

  const healthTips = useMemo((): HealthTip[] => {
    if (!averages) return [];

    const tips: HealthTip[] = [];

    // Sleep tips
    if (averages.sleep < 6) {
      tips.push({
        id: 'sleep-critical',
        category: 'sleep',
        priority: 'high',
        title: 'Prioritize Your Sleep',
        description: 'You\'re averaging less than 6 hours of sleep. Aim for 7-9 hours to improve cognitive function, immune health, and overall well-being.',
        icon: '😴',
      });
    } else if (averages.sleep < 7) {
      tips.push({
        id: 'sleep-improve',
        category: 'sleep',
        priority: 'medium',
        title: 'Improve Sleep Quality',
        description: 'Try to get 30 more minutes of sleep. Avoid screens 1 hour before bed and keep your room cool and dark.',
        icon: '🌙',
      });
    } else {
      tips.push({
        id: 'sleep-maintain',
        category: 'sleep',
        priority: 'low',
        title: 'Great Sleep Habits!',
        description: 'You\'re getting adequate sleep. Maintain your routine for continued benefits.',
        icon: '✨',
      });
    }

    // Exercise tips
    if (averages.exercise < 15) {
      tips.push({
        id: 'exercise-critical',
        category: 'exercise',
        priority: 'high',
        title: 'Increase Physical Activity',
        description: 'Start with short 10-minute walks and gradually increase. Even light activity improves heart health and mood.',
        icon: '🏃',
      });
    } else if (averages.exercise < 30) {
      tips.push({
        id: 'exercise-improve',
        category: 'exercise',
        priority: 'medium',
        title: 'Boost Your Activity Level',
        description: 'You\'re on the right track! Try adding 15 more minutes of moderate exercise daily.',
        icon: '💪',
      });
    } else {
      tips.push({
        id: 'exercise-maintain',
        category: 'exercise',
        priority: 'low',
        title: 'Excellent Activity Level!',
        description: 'You\'re meeting exercise recommendations. Keep up the great work!',
        icon: '🏆',
      });
    }

    // Stress tips
    if (averages.stress > 7) {
      tips.push({
        id: 'stress-critical',
        category: 'stress',
        priority: 'high',
        title: 'Manage Your Stress',
        description: 'High stress levels detected. Try deep breathing exercises, meditation, or speaking with a mental health professional.',
        icon: '🧘',
      });
    } else if (averages.stress > 5) {
      tips.push({
        id: 'stress-moderate',
        category: 'stress',
        priority: 'medium',
        title: 'Stress Management Tips',
        description: 'Consider adding relaxation techniques to your routine: yoga, journaling, or nature walks can help.',
        icon: '🌿',
      });
    }

    // Diet tips
    if (averages.diet < 5) {
      tips.push({
        id: 'diet-improve',
        category: 'diet',
        priority: 'high',
        title: 'Improve Your Diet',
        description: 'Focus on adding more fruits, vegetables, and whole grains. Small changes can make a big difference.',
        icon: '🥗',
      });
    } else if (averages.diet < 7) {
      tips.push({
        id: 'diet-enhance',
        category: 'diet',
        priority: 'medium',
        title: 'Enhance Your Nutrition',
        description: 'Great progress! Consider reducing processed foods and adding more colorful vegetables to your meals.',
        icon: '🍎',
      });
    }

    // Steps tips
    if (averages.steps < 5000) {
      tips.push({
        id: 'steps-increase',
        category: 'general',
        priority: 'medium',
        title: 'Move More Throughout the Day',
        description: 'Take short walking breaks every hour. Aim for at least 5,000 steps to start, working up to 10,000.',
        icon: '👟',
      });
    }

    // Hydration reminder
    tips.push({
      id: 'hydration-reminder',
      category: 'hydration',
      priority: 'low',
      title: 'Stay Hydrated',
      description: 'Drink at least 8 glasses of water daily. Proper hydration improves energy, skin health, and digestion.',
      icon: '💧',
    });

    return tips.sort((a, b) => {
      const priorityOrder = { high: 0, medium: 1, low: 2 };
      return priorityOrder[a.priority] - priorityOrder[b.priority];
    });
  }, [averages]);

  const healthAlerts = useMemo((): HealthAlert[] => {
    if (!averages || !logs) return [];

    const alerts: HealthAlert[] = [];

    // Check for concerning patterns
    if (averages.sleep < 5) {
      alerts.push({
        id: 'sleep-warning',
        severity: 'warning',
        message: 'Your sleep is critically low. Chronic sleep deprivation can affect health. Consider consulting a doctor if this persists.',
        action: 'Speak to a healthcare provider',
      });
    }

    if (averages.stress > 8) {
      alerts.push({
        id: 'stress-warning',
        severity: 'warning',
        message: 'Very high stress levels detected. Please consider speaking with a mental health professional.',
        action: 'Seek professional support',
      });
    }

    // Positive alerts
    if (averages.exercise >= 30 && averages.sleep >= 7 && averages.diet >= 7) {
      alerts.push({
        id: 'great-health',
        severity: 'success',
        message: 'Your health metrics are excellent! Keep maintaining these healthy habits.',
      });
    }

    return alerts;
  }, [averages, logs]);

  const overallHealthScore = useMemo(() => {
    if (!averages) return 50;

    let score = 50;

    // Sleep score (max 20 points)
    if (averages.sleep >= 7 && averages.sleep <= 9) score += 20;
    else if (averages.sleep >= 6) score += 12;
    else score += 5;

    // Exercise score (max 15 points)
    if (averages.exercise >= 30) score += 15;
    else if (averages.exercise >= 15) score += 10;
    else score += 3;

    // Diet score (max 10 points)
    score += (averages.diet / 10) * 10;

    // Stress score (max 10 points) - lower is better
    score += ((10 - averages.stress) / 10) * 10;

    // Steps score (max 5 points)
    if (averages.steps >= 10000) score += 5;
    else if (averages.steps >= 5000) score += 3;

    return Math.min(100, Math.max(0, Math.round(score)));
  }, [averages]);

  return {
    healthTips,
    healthAlerts,
    overallHealthScore,
    averages,
    isLoading,
  };
}
