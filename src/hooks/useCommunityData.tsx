import { useQuery } from '@tanstack/react-query';
import { supabase } from '@/integrations/supabase/client';

export interface CommunityHealthData {
  id: string;
  region: string;
  date: string;
  avg_health_score: number | null;
  common_symptoms: string[] | null;
  total_users: number | null;
  risk_distribution: {
    low: number;
    medium: number;
    high: number;
  } | null;
  created_at: string;
}

export function useCommunityData() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['community-health-data'],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('community_health_data')
        .select('*')
        .order('date', { ascending: true });
      
      if (error) throw error;
      return data as CommunityHealthData[];
    },
  });

  // Group data by region
  const byRegion = data?.reduce((acc, item) => {
    if (!acc[item.region]) {
      acc[item.region] = [];
    }
    acc[item.region].push(item);
    return acc;
  }, {} as Record<string, CommunityHealthData[]>) || {};

  // Get latest data per region
  const latestByRegion = Object.entries(byRegion).reduce((acc, [region, items]) => {
    acc[region] = items[items.length - 1];
    return acc;
  }, {} as Record<string, CommunityHealthData>);

  // Calculate overall stats
  const overallStats = {
    avgHealthScore: data?.length
      ? data.reduce((acc, item) => acc + (item.avg_health_score || 0), 0) / data.length
      : 0,
    totalUsers: Object.values(latestByRegion).reduce(
      (acc, item) => acc + (item?.total_users || 0),
      0
    ),
    regions: Object.keys(byRegion).length,
  };

  return {
    data,
    byRegion,
    latestByRegion,
    overallStats,
    isLoading,
    error,
  };
}