import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './useAuth';
import { useToast } from './use-toast';

export interface LifestyleLog {
  id: string;
  user_id: string;
  log_date: string;
  sleep_hours: number | null;
  exercise_minutes: number | null;
  daily_steps: number | null;
  diet_quality: number | null;
  stress_level: number | null;
  water_glasses: number | null;
  smoking: boolean;
  alcohol_units: number;
  notes: string | null;
  created_at: string;
}

export interface NewLifestyleLog {
  log_date?: string;
  sleep_hours?: number;
  exercise_minutes?: number;
  daily_steps?: number;
  diet_quality?: number;
  stress_level?: number;
  water_glasses?: number;
  smoking?: boolean;
  alcohol_units?: number;
  notes?: string;
}

export function useLifestyleLogs() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: logs, isLoading, error } = useQuery({
    queryKey: ['lifestyle-logs', user?.id],
    queryFn: async () => {
      if (!user) return [];
      const { data, error } = await supabase
        .from('lifestyle_logs')
        .select('*')
        .eq('user_id', user.id)
        .order('log_date', { ascending: false })
        .limit(30);
      
      if (error) throw error;
      return data as LifestyleLog[];
    },
    enabled: !!user,
  });

  const addLog = useMutation({
    mutationFn: async (log: NewLifestyleLog) => {
      if (!user) throw new Error('Not authenticated');
      
      const logDate = log.log_date || new Date().toISOString().split('T')[0];
      
      // Check if log exists for this date
      const { data: existing } = await supabase
        .from('lifestyle_logs')
        .select('id')
        .eq('user_id', user.id)
        .eq('log_date', logDate)
        .maybeSingle();
      
      if (existing) {
        // Update existing log
        const { error } = await supabase
          .from('lifestyle_logs')
          .update({ ...log, log_date: logDate })
          .eq('id', existing.id);
        if (error) throw error;
      } else {
        // Insert new log
        const { error } = await supabase
          .from('lifestyle_logs')
          .insert({ ...log, log_date: logDate, user_id: user.id });
        if (error) throw error;
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['lifestyle-logs', user?.id] });
      toast({
        title: 'Log saved',
        description: 'Your lifestyle data has been recorded.',
      });
    },
    onError: (error) => {
      toast({
        title: 'Error',
        description: error.message,
        variant: 'destructive',
      });
    },
  });

  const todayLog = logs?.find(
    (log) => log.log_date === new Date().toISOString().split('T')[0]
  );

  // Calculate averages for the last 7 days
  const last7Days = logs?.slice(0, 7) || [];
  const averages = {
    sleep: last7Days.length > 0
      ? last7Days.reduce((acc, log) => acc + (log.sleep_hours || 0), 0) / last7Days.length
      : 0,
    exercise: last7Days.length > 0
      ? last7Days.reduce((acc, log) => acc + (log.exercise_minutes || 0), 0) / last7Days.length
      : 0,
    steps: last7Days.length > 0
      ? last7Days.reduce((acc, log) => acc + (log.daily_steps || 0), 0) / last7Days.length
      : 0,
    diet: last7Days.length > 0
      ? last7Days.reduce((acc, log) => acc + (log.diet_quality || 0), 0) / last7Days.length
      : 0,
    stress: last7Days.length > 0
      ? last7Days.reduce((acc, log) => acc + (log.stress_level || 0), 0) / last7Days.length
      : 0,
  };

  return {
    logs,
    isLoading,
    error,
    addLog: addLog.mutate,
    isAdding: addLog.isPending,
    todayLog,
    averages,
  };
}