import { MainLayout } from '@/components/layout/MainLayout';
import { useCommunityData } from '@/hooks/useCommunityData';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { StatCard } from '@/components/ui/stat-card';
import { Skeleton } from '@/components/ui/skeleton';
import { Users, MapPin, TrendingUp, Activity } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';

const COLORS = ['hsl(152, 60%, 45%)', 'hsl(38, 92%, 50%)', 'hsl(0, 72%, 55%)'];

export default function CommunityPage() {
  const { byRegion, latestByRegion, overallStats, isLoading } = useCommunityData();

  if (isLoading) {
    return (
      <MainLayout>
        <div className="space-y-6">
          <Skeleton className="h-10 w-64" />
          <div className="grid gap-4 md:grid-cols-3"><Skeleton className="h-32" /><Skeleton className="h-32" /><Skeleton className="h-32" /></div>
        </div>
      </MainLayout>
    );
  }

  const regionTrendData = Object.entries(byRegion).map(([region, data]) => ({
    region,
    score: data[data.length - 1]?.avg_health_score || 0,
  }));

  return (
    <MainLayout>
      <div className="space-y-8 animate-fade-in">
        <div>
          <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Community Health Trends</h1>
          <p className="text-muted-foreground">Anonymized health analytics across regions</p>
        </div>

        {/* Overview Stats */}
        <div className="grid gap-4 sm:grid-cols-3">
          <StatCard title="Total Users" value={overallStats.totalUsers.toLocaleString()} icon={<Users className="h-5 w-5" />} variant="primary" />
          <StatCard title="Regions Tracked" value={overallStats.regions} icon={<MapPin className="h-5 w-5" />} variant="success" />
          <StatCard title="Avg Health Score" value={overallStats.avgHealthScore.toFixed(1)} icon={<TrendingUp className="h-5 w-5" />} variant="default" />
        </div>

        {/* Regional Comparison */}
        <Card className="shadow-card border-border/50">
          <CardHeader>
            <CardTitle>Regional Health Scores</CardTitle>
            <CardDescription>Average health scores by region</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={regionTrendData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="region" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                  <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px' }} />
                  <Bar dataKey="score" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} name="Health Score" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>

        {/* Region Cards */}
        <div className="grid gap-4 sm:grid-cols-2">
          {Object.entries(latestByRegion).map(([region, data]) => {
            const riskDist = data.risk_distribution as { low: number; medium: number; high: number } | null;
            const pieData = riskDist ? [
              { name: 'Low', value: riskDist.low },
              { name: 'Medium', value: riskDist.medium },
              { name: 'High', value: riskDist.high },
            ] : [];

            return (
              <Card key={region} className="shadow-card border-border/50">
                <CardHeader className="pb-2">
                  <CardTitle className="flex items-center gap-2 text-base">
                    <MapPin className="h-4 w-4 text-primary" />
                    {region}
                  </CardTitle>
                  <CardDescription>{data.total_users?.toLocaleString()} users</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-6">
                    <div className="h-24 w-24">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <Pie data={pieData} dataKey="value" cx="50%" cy="50%" innerRadius={25} outerRadius={40}>
                            {pieData.map((_, i) => <Cell key={i} fill={COLORS[i]} />)}
                          </Pie>
                        </PieChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="flex-1 space-y-1 text-sm">
                      <div className="flex justify-between"><span className="text-muted-foreground">Avg Score</span><span className="font-medium">{data.avg_health_score}</span></div>
                      <div className="flex justify-between"><span className="text-success">Low Risk</span><span>{riskDist?.low}%</span></div>
                      <div className="flex justify-between"><span className="text-warning">Medium</span><span>{riskDist?.medium}%</span></div>
                      <div className="flex justify-between"><span className="text-destructive">High Risk</span><span>{riskDist?.high}%</span></div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </MainLayout>
  );
}