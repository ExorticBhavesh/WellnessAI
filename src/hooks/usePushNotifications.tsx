import { useState, useEffect, useCallback } from 'react';
import { useAuth } from './useAuth';
import { useToast } from './use-toast';

interface ScheduledReminder {
  id: string;
  title: string;
  body: string;
  time: string; // HH:MM format
  enabled: boolean;
  type: 'health' | 'medication';
}

export function usePushNotifications() {
  const { user } = useAuth();
  const { toast } = useToast();
  const [permission, setPermission] = useState<NotificationPermission>('default');
  const [reminders, setReminders] = useState<ScheduledReminder[]>(() => {
    const stored = localStorage.getItem('health-reminders');
    return stored ? JSON.parse(stored) : getDefaultReminders();
  });

  useEffect(() => {
    if ('Notification' in window) {
      setPermission(Notification.permission);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('health-reminders', JSON.stringify(reminders));
  }, [reminders]);

  const requestPermission = useCallback(async () => {
    if (!('Notification' in window)) {
      toast({
        title: 'Notifications not supported',
        description: 'Your browser does not support push notifications.',
        variant: 'destructive',
      });
      return false;
    }

    try {
      const result = await Notification.requestPermission();
      setPermission(result);
      
      if (result === 'granted') {
        toast({
          title: 'Notifications enabled',
          description: 'You will now receive health reminders.',
        });
        return true;
      } else {
        toast({
          title: 'Notifications blocked',
          description: 'Please enable notifications in your browser settings.',
          variant: 'destructive',
        });
        return false;
      }
    } catch (error) {
      console.error('Error requesting notification permission:', error);
      return false;
    }
  }, [toast]);

  const sendNotification = useCallback((title: string, body: string, icon?: string) => {
    if (permission !== 'granted') return;

    try {
      new Notification(title, {
        body,
        icon: icon || '/favicon.ico',
        badge: '/favicon.ico',
        tag: `health-reminder-${Date.now()}`,
      });
    } catch (error) {
      console.error('Error sending notification:', error);
    }
  }, [permission]);

  const addReminder = useCallback((reminder: Omit<ScheduledReminder, 'id'>) => {
    const newReminder: ScheduledReminder = {
      ...reminder,
      id: crypto.randomUUID(),
    };
    setReminders(prev => [...prev, newReminder]);
    toast({
      title: 'Reminder added',
      description: `${reminder.title} scheduled for ${reminder.time}`,
    });
  }, [toast]);

  const updateReminder = useCallback((id: string, updates: Partial<ScheduledReminder>) => {
    setReminders(prev => prev.map(r => r.id === id ? { ...r, ...updates } : r));
  }, []);

  const deleteReminder = useCallback((id: string) => {
    setReminders(prev => prev.filter(r => r.id !== id));
    toast({
      title: 'Reminder deleted',
    });
  }, [toast]);

  const toggleReminder = useCallback((id: string) => {
    setReminders(prev => prev.map(r => 
      r.id === id ? { ...r, enabled: !r.enabled } : r
    ));
  }, []);

  // Check for due reminders every minute
  useEffect(() => {
    if (permission !== 'granted') return;

    const checkReminders = () => {
      const now = new Date();
      const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      
      reminders
        .filter(r => r.enabled && r.time === currentTime)
        .forEach(reminder => {
          sendNotification(reminder.title, reminder.body);
        });
    };

    const interval = setInterval(checkReminders, 60000);
    return () => clearInterval(interval);
  }, [permission, reminders, sendNotification]);

  return {
    permission,
    isSupported: 'Notification' in window,
    reminders,
    requestPermission,
    sendNotification,
    addReminder,
    updateReminder,
    deleteReminder,
    toggleReminder,
  };
}

function getDefaultReminders(): ScheduledReminder[] {
  return [
    {
      id: crypto.randomUUID(),
      title: 'Morning Health Check',
      body: 'Time to log your morning vitals and how you\'re feeling today!',
      time: '08:00',
      enabled: false,
      type: 'health',
    },
    {
      id: crypto.randomUUID(),
      title: 'Hydration Reminder',
      body: 'Don\'t forget to drink water! Stay hydrated for better health.',
      time: '12:00',
      enabled: false,
      type: 'health',
    },
    {
      id: crypto.randomUUID(),
      title: 'Evening Wellness Log',
      body: 'Log your daily exercise, diet, and stress levels before bed.',
      time: '20:00',
      enabled: false,
      type: 'health',
    },
  ];
}
