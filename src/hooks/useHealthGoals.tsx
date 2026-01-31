import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './useAuth';
import { useToast } from './use-toast';

export interface HealthGoal {
  id: string;
  user_id: string;
  goal_type: string;
  goal_name: string;
  target_value: number;
  unit: string;
  current_value: number;
  start_date: string;
  end_date: string | null;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export interface NewHealthGoal {
  goal_type: string;
  goal_name: string;
  target_value: number;
  unit: string;
  end_date?: string;
}

export interface UpdateHealthGoal {
  id: string;
  current_value?: number;
  target_value?: number;
  is_active?: boolean;
}

export const GOAL_PRESETS = [
  { type: 'steps', name: 'Daily Steps', target: 10000, unit: 'steps', icon: 'Footprints' },
  { type: 'sleep', name: 'Sleep Hours', target: 8, unit: 'hours', icon: 'Moon' },
  { type: 'water', name: 'Water Intake', target: 8, unit: 'glasses', icon: 'Droplets' },
  { type: 'exercise', name: 'Exercise Time', target: 30, unit: 'minutes', icon: 'Dumbbell' },
  { type: 'weight', name: 'Target Weight', target: 70, unit: 'kg', icon: 'Scale' },
] as const;

export function useHealthGoals() {
  const { user } = useAuth();
  const { toast } = useToast();
  const queryClient = useQueryClient();

  const { data: goals, isLoading, error } = useQuery({
    queryKey: ['health-goals', user?.id],
    queryFn: async () => {
      if (!user) return [];
      const { data, error } = await supabase
        .from('health_goals')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      return data as HealthGoal[];
    },
    enabled: !!user,
  });

  const addGoal = useMutation({
    mutationFn: async (goal: NewHealthGoal) => {
      if (!user) throw new Error('Not authenticated');
      
      const { error } = await supabase
        .from('health_goals')
        .insert({
          ...goal,
          user_id: user.id,
          current_value: 0,
        });
      
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['health-goals', user?.id] });
      toast({
        title: 'Goal created',
        description: 'Your health goal has been set successfully.',
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

  const updateGoal = useMutation({
    mutationFn: async ({ id, ...updates }: UpdateHealthGoal) => {
      if (!user) throw new Error('Not authenticated');
      
      const { error } = await supabase
        .from('health_goals')
        .update(updates)
        .eq('id', id)
        .eq('user_id', user.id);
      
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['health-goals', user?.id] });
      toast({
        title: 'Goal updated',
        description: 'Your progress has been saved.',
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

  const deleteGoal = useMutation({
    mutationFn: async (goalId: string) => {
      if (!user) throw new Error('Not authenticated');
      
      const { error } = await supabase
        .from('health_goals')
        .delete()
        .eq('id', goalId)
        .eq('user_id', user.id);
      
      if (error) throw error;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['health-goals', user?.id] });
      toast({
        title: 'Goal deleted',
        description: 'Your health goal has been removed.',
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

  const activeGoals = goals?.filter(g => g.is_active) || [];
  const completedGoals = goals?.filter(g => !g.is_active) || [];

  const getProgress = (goal: HealthGoal) => {
    return Math.min((goal.current_value / goal.target_value) * 100, 100);
  };

  return {
    goals,
    activeGoals,
    completedGoals,
    isLoading,
    error,
    addGoal: addGoal.mutate,
    updateGoal: updateGoal.mutate,
    deleteGoal: deleteGoal.mutate,
    isAdding: addGoal.isPending,
    isUpdating: updateGoal.isPending,
    getProgress,
  };
}
