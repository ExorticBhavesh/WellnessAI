import { useNavigate } from 'react-router-dom';
import { useAuth } from '@/hooks/useAuth';
import { useProfile } from '@/hooks/useProfile';
import { useLifestyleLogs } from '@/hooks/useLifestyleLogs';
import { MainLayout } from '@/components/layout/MainLayout';
import { HealthScoreRing } from '@/components/ui/health-score-ring';
import { StatCard } from '@/components/ui/stat-card';
import { HealthReportGenerator } from '@/components/health-report/HealthReportGenerator';
import { HealthTipsCard } from '@/components/dashboard/HealthTipsCard';
import { HealthGoalsCard } from '@/components/goals/HealthGoalsCard';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import {
  Activity,
  Moon,
  Footprints,
  Apple,
  Brain,
  Droplets,
  ArrowRight,
  Stethoscope,
  BarChart3,
  Sparkles,
  TrendingUp,
  Calendar,
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
} from 'recharts';

export default function Dashboard() {
  const navigate = useNavigate();
  const { user } = useAuth();
  const { profile, isLoading: profileLoading } = useProfile();
  const { logs, averages, isLoading: logsLoading } = useLifestyleLogs();

  // Calculate health score based on lifestyle averages
  const calculateHealthScore = () => {
    if (!averages) return 65;
    
    let score = 50;
    
    // Sleep score (7-9 hours optimal)
    if (averages.sleep >= 7 && averages.sleep <= 9) score += 15;
    else if (averages.sleep >= 6) score += 10;
    else score += 5;
    
    // Exercise score
    if (averages.exercise >= 30) score += 15;
    else if (averages.exercise >= 15) score += 10;
    else score += 3;
    
    // Steps score
    if (averages.steps >= 10000) score += 10;
    else if (averages.steps >= 5000) score += 6;
    else score += 2;
    
    // Diet score
    score += (averages.diet / 10) * 10;
    
    // Stress (lower is better)
    score += ((10 - averages.stress) / 10) * 10;
    
    return Math.min(100, Math.max(0, score));
  };

  const healthScore = calculateHealthScore();

  // Prepare chart data
  const chartData = logs?.slice(0, 7).reverse().map((log) => ({
    date: new Date(log.log_date).toLocaleDateString('en-US', { weekday: 'short' }),
    sleep: log.sleep_hours || 0,
    exercise: log.exercise_minutes || 0,
    steps: (log.daily_steps || 0) / 1000,
    stress: log.stress_level || 0,
  })) || [];

  const greeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 18) return 'Good afternoon';
    return 'Good evening';
  };

  const firstName = profile?.full_name?.split(' ')[0] || user?.email?.split('@')[0] || 'there';

  if (profileLoading || logsLoading) {
    return (
      <MainLayout>
        <div className="space-y-6">
          <Skeleton className="h-12 w-80" />
          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-36 rounded-2xl" />
            ))}
          </div>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout>
      <div className="space-y-8 animate-fade-in">
        {/* Header */}
        <div className="flex flex-col gap-6 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-3xl font-bold tracking-tight sm:text-4xl mb-2">
              {greeting()}, <span className="gradient-text">{firstName}</span> 👋
            </h1>
            <p className="text-muted-foreground text-lg">
              Here's your wellness overview for today
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <HealthReportGenerator />
            <Button variant="outline" onClick={() => navigate('/lifestyle')} className="gap-2 hover-lift">
              <Calendar className="h-4 w-4" />
              Log Today
            </Button>
            <Button className="gradient-health gap-2 shadow-lg shadow-primary/25 hover:shadow-xl hover:shadow-primary/30 hover:scale-105 transition-all" onClick={() => navigate('/symptoms')}>
              <Stethoscope className="h-4 w-4" />
              Check Symptoms
            </Button>
          </div>
        </div>

        {/* Health Score & Quick Stats */}
        <div className="grid gap-6 lg:grid-cols-3">
          {/* Health Score Card */}
          <Card className="lg:row-span-2 shadow-card border-border/40 overflow-hidden relative group hover:shadow-card-hover transition-all duration-300">
            <div className="absolute inset-0 gradient-mesh opacity-5 group-hover:opacity-10 transition-opacity" />
            <CardHeader className="text-center relative">
              <CardTitle className="text-xl">Your Health Score</CardTitle>
              <CardDescription>Based on your lifestyle data</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col items-center gap-6 relative">
              <div className="relative">
                <div className="absolute inset-0 rounded-full bg-primary/10 blur-xl scale-110 animate-pulse-slow" />
                <HealthScoreRing score={healthScore} size="xl" label="Overall Score" />
              </div>
              <div className="w-full space-y-4 bg-muted/30 rounded-2xl p-4">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <Moon className="h-4 w-4 text-primary" />
                    <span className="text-muted-foreground">Sleep Quality</span>
                  </div>
                  <span className="font-semibold">{averages.sleep.toFixed(1)}h avg</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <Activity className="h-4 w-4 text-success" />
                    <span className="text-muted-foreground">Activity Level</span>
                  </div>
                  <span className="font-semibold">{averages.exercise.toFixed(0)} min/day</span>
                </div>
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center gap-2">
                    <Brain className="h-4 w-4 text-warning" />
                    <span className="text-muted-foreground">Stress Level</span>
                  </div>
                  <span className="font-semibold">{averages.stress.toFixed(1)}/10</span>
                </div>
              </div>
              <Button variant="outline" className="w-full gap-2 hover-lift" onClick={() => navigate('/profile')}>
                <Sparkles className="h-4 w-4" />
                Get AI Insights
              </Button>
            </CardContent>
          </Card>

          {/* Quick Stats Grid */}
          <div className="grid gap-4 sm:grid-cols-2 lg:col-span-2">
            <StatCard
              title="Sleep"
              value={averages.sleep.toFixed(1)}
              unit="hours"
              icon={<Moon className="h-5 w-5" />}
              variant={averages.sleep >= 7 ? 'success' : averages.sleep >= 6 ? 'warning' : 'danger'}
              description="7-day average"
              trend={averages.sleep >= 7 ? 'up' : 'down'}
              trendValue={averages.sleep >= 7 ? 'Good sleep!' : 'Need more rest'}
            />
            <StatCard
              title="Exercise"
              value={averages.exercise.toFixed(0)}
              unit="min"
              icon={<Activity className="h-5 w-5" />}
              variant={averages.exercise >= 30 ? 'success' : averages.exercise >= 15 ? 'warning' : 'danger'}
              description="Daily average"
              trend={averages.exercise >= 30 ? 'up' : 'neutral'}
              trendValue={averages.exercise >= 30 ? 'On target' : 'Keep moving'}
            />
            <StatCard
              title="Daily Steps"
              value={(averages.steps / 1000).toFixed(1)}
              unit="K"
              icon={<Footprints className="h-5 w-5" />}
              variant={averages.steps >= 10000 ? 'success' : averages.steps >= 5000 ? 'warning' : 'default'}
              description="7-day average"
            />
            <StatCard
              title="Diet Quality"
              value={averages.diet.toFixed(1)}
              unit="/10"
              icon={<Apple className="h-5 w-5" />}
              variant={averages.diet >= 7 ? 'success' : averages.diet >= 5 ? 'warning' : 'danger'}
              description="Weekly score"
            />
            <StatCard
              title="Stress Level"
              value={averages.stress.toFixed(1)}
              unit="/10"
              icon={<Brain className="h-5 w-5" />}
              variant={averages.stress <= 4 ? 'success' : averages.stress <= 6 ? 'warning' : 'danger'}
              description="Lower is better"
            />
            <StatCard
              title="Hydration"
              value="6"
              unit="glasses"
              icon={<Droplets className="h-5 w-5" />}
              variant="primary"
              description="Today's intake"
            />
          </div>
        </div>

        {/* Charts */}
        <div className="grid gap-6 lg:grid-cols-2">
          {/* Sleep & Exercise Trend */}
          <Card className="shadow-card border-border/40 hover:shadow-card-hover transition-all duration-300">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <TrendingUp className="h-5 w-5 text-primary" />
                    Weekly Activity Trend
                  </CardTitle>
                  <CardDescription>Sleep hours and exercise minutes</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="sleepGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--chart-1))" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="hsl(var(--chart-1))" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="exerciseGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="hsl(var(--chart-2))" stopOpacity={0.4} />
                        <stop offset="95%" stopColor="hsl(var(--chart-2))" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '12px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="sleep"
                      stroke="hsl(var(--chart-1))"
                      fill="url(#sleepGradient)"
                      strokeWidth={3}
                      name="Sleep (hrs)"
                    />
                    <Area
                      type="monotone"
                      dataKey="exercise"
                      stroke="hsl(var(--chart-2))"
                      fill="url(#exerciseGradient)"
                      strokeWidth={3}
                      name="Exercise (min)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>

          {/* Steps Trend */}
          <Card className="shadow-card border-border/40 hover:shadow-card-hover transition-all duration-300">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div>
                  <CardTitle className="text-lg flex items-center gap-2">
                    <Footprints className="h-5 w-5 text-success" />
                    Daily Steps
                  </CardTitle>
                  <CardDescription>Your step count over the week</CardDescription>
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <YAxis stroke="hsl(var(--muted-foreground))" fontSize={12} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: 'hsl(var(--card))',
                        border: '1px solid hsl(var(--border))',
                        borderRadius: '12px',
                        boxShadow: '0 4px 12px rgba(0,0,0,0.1)',
                      }}
                      formatter={(value: number) => [`${value}K steps`, 'Steps']}
                    />
                    <Line
                      type="monotone"
                      dataKey="steps"
                      stroke="hsl(var(--chart-3))"
                      strokeWidth={3}
                      dot={{ fill: 'hsl(var(--chart-3))', strokeWidth: 2, r: 5 }}
                      activeDot={{ r: 7, stroke: 'hsl(var(--background))', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Health Goals & Tips Section */}
        <div className="grid gap-6 lg:grid-cols-2">
          <HealthGoalsCard />
          <HealthTipsCard />
        </div>

        {/* AI Health Assistant */}
        <div className="grid gap-6 lg:grid-cols-1">
          {/* AI Health Assistant Promo */}
          
          {/* AI Health Assistant Promo */}
          <Card className="shadow-card border-border/40 hover:shadow-card-hover transition-all duration-300 overflow-hidden relative group">
            <div className="absolute inset-0 gradient-mesh opacity-10 group-hover:opacity-20 transition-opacity" />
            <CardHeader className="relative">
              <CardTitle className="flex items-center gap-2">
                <Stethoscope className="h-5 w-5 text-primary" />
                AI Health Assistant
              </CardTitle>
              <CardDescription>
                Get instant health guidance with our AI chatbot
              </CardDescription>
            </CardHeader>
            <CardContent className="relative">
              <div className="space-y-4">
                <p className="text-sm text-muted-foreground">
                  Our AI health assistant can help you:
                </p>
                <ul className="text-sm space-y-2">
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Understand your symptoms
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Get personalized health tips
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Answer wellness questions
                  </li>
                  <li className="flex items-center gap-2">
                    <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                    Voice responses for accessibility
                  </li>
                </ul>
                <div className="bg-warning/10 border border-warning/30 rounded-lg p-3 mt-4">
                  <p className="text-xs text-warning-foreground flex items-start gap-2">
                    <span className="text-warning">⚠️</span>
                    <span>
                      Our AI always advises consulting a healthcare professional for severe symptoms or persistent health concerns.
                    </span>
                  </p>
                </div>
                <p className="text-xs text-muted-foreground mt-2">
                  Click the chat icon in the bottom-right corner to start a conversation.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Quick Actions */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <Card className="interactive-card group" onClick={() => navigate('/symptoms')}>
            <CardContent className="flex items-center gap-4 p-6">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl gradient-health shadow-lg group-hover:scale-110 transition-transform">
                <Stethoscope className="h-7 w-7 text-primary-foreground" />
              </div>
              <div className="flex-1">
                <h3 className="font-bold text-lg">AI Symptom Check</h3>
                <p className="text-sm text-muted-foreground">Get instant health insights</p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground transition-all group-hover:translate-x-2 group-hover:text-primary" />
            </CardContent>
          </Card>

          <Card className="interactive-card group" onClick={() => navigate('/community')}>
            <CardContent className="flex items-center gap-4 p-6">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl gradient-energy shadow-lg group-hover:scale-110 transition-transform">
                <BarChart3 className="h-7 w-7 text-primary-foreground" />
              </div>
              <div className="flex-1">
                <h3 className="font-bold text-lg">Community Trends</h3>
                <p className="text-sm text-muted-foreground">See regional health data</p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground transition-all group-hover:translate-x-2 group-hover:text-primary" />
            </CardContent>
          </Card>

          <Card className="interactive-card group sm:col-span-2 lg:col-span-1" onClick={() => navigate('/lifestyle')}>
            <CardContent className="flex items-center gap-4 p-6">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl gradient-wellness shadow-lg group-hover:scale-110 transition-transform">
                <Activity className="h-7 w-7 text-primary-foreground" />
              </div>
              <div className="flex-1">
                <h3 className="font-bold text-lg">Track Lifestyle</h3>
                <p className="text-sm text-muted-foreground">Log your daily habits</p>
              </div>
              <ArrowRight className="h-5 w-5 text-muted-foreground transition-all group-hover:translate-x-2 group-hover:text-primary" />
            </CardContent>
          </Card>
        </div>
      </div>
    </MainLayout>
  );
}
