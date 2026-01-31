import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';
import { useAuth } from './useAuth';

export interface SymptomCheck {
  id: string;
  user_id: string;
  symptoms: string[];
  risk_level: 'low' | 'medium' | 'high' | null;
  risk_score: number | null;
  ai_analysis: string | null;
  recommendations: string[] | null;
  created_at: string;
}

export function useSymptomHistory() {
  const { user } = useAuth();

  const { data: history, isLoading, error, refetch } = useQuery({
    queryKey: ['symptom-history', user?.id],
    queryFn: async () => {
      if (!user) return [];
      
      const { data, error } = await supabase
        .from('symptom_checks')
        .select('*')
        .eq('user_id', user.id)
        .order('created_at', { ascending: false });

      if (error) throw error;
      return data as SymptomCheck[];
    },
    enabled: !!user,
  });

  return {
    history: history || [],
    isLoading,
    error,
    refetch,
  };
}
