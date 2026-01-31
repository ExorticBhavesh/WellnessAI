import { useState } from 'react';
import { MainLayout } from '@/components/layout/MainLayout';
import { HealthGoalsCard } from '@/components/goals/HealthGoalsCard';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Button } from '@/components/ui/button';
import { useHealthGoals, GOAL_PRESETS, HealthGoal } from '@/hooks/useHealthGoals';
import { Target, Trophy, TrendingUp, CheckCircle2, Calendar, BarChart3 } from 'lucide-react';
import { cn } from '@/lib/utils';

export default function Goals() {
  const { goals, activeGoals, completedGoals, isLoading, getProgress } = useHealthGoals();
  const [selectedTab, setSelectedTab] = useState('active');

  // Calculate overall progress
  const overallProgress = activeGoals.length > 0
    ? activeGoals.reduce((acc, goal) => acc + getProgress(goal), 0) / activeGoals.length
    : 0;

  // Calculate stats
  const totalGoalsCreated = goals?.length || 0;
  const goalsCompleted = completedGoals.length;
  const currentStreak = 7; // Placeholder - could calculate from actual data

  return (
    <MainLayout>
      <div className="space-y-8 animate-fade-in">
        {/* Header */}
        <div className="flex flex-col gap-4">
          <div>
            <h1 className="text-3xl font-bold tracking-tight sm:text-4xl mb-2">
              Health <span className="gradient-text">Goals</span> 🎯
            </h1>
            <p className="text-muted-foreground text-lg">
              Set and track your wellness objectives
            </p>
          </div>
        </div>

        {/* Stats Overview */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Card className="stat-card">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Active Goals</p>
                  <p className="text-3xl font-bold">{activeGoals.length}</p>
                </div>
                <div className="p-3 rounded-xl bg-primary/10">
                  <Target className="h-6 w-6 text-primary" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="stat-card">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Completed</p>
                  <p className="text-3xl font-bold">{goalsCompleted}</p>
                </div>
                <div className="p-3 rounded-xl bg-success/10">
                  <Trophy className="h-6 w-6 text-success" />
                </div>
              </div>
            </CardContent>
          </Card>

          <Card className="stat-card">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Overall Progress</p>
                  <p className="text-3xl font-bold">{overallProgress.toFixed(0)}%</p>
                </div>
                <div className="p-3 rounded-xl bg-info/10">
                  <TrendingUp className="h-6 w-6 text-info" />
                </div>
              </div>
              <Progress value={overallProgress} className="mt-3 h-2" />
            </CardContent>
          </Card>

          <Card className="stat-card">
            <CardContent className="pt-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-muted-foreground">Current Streak</p>
                  <p className="text-3xl font-bold">{currentStreak} days</p>
                </div>
                <div className="p-3 rounded-xl bg-warning/10">
                  <Calendar className="h-6 w-6 text-warning" />
                </div>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Goals Tabs */}
        <Tabs value={selectedTab} onValueChange={setSelectedTab} className="space-y-6">
          <TabsList className="grid w-full max-w-md grid-cols-2">
            <TabsTrigger value="active" className="gap-2">
              <Target className="h-4 w-4" />
              Active ({activeGoals.length})
            </TabsTrigger>
            <TabsTrigger value="completed" className="gap-2">
              <CheckCircle2 className="h-4 w-4" />
              Completed ({completedGoals.length})
            </TabsTrigger>
          </TabsList>

          <TabsContent value="active" className="space-y-6">
            <div className="grid gap-6 lg:grid-cols-2">
              <HealthGoalsCard />
              
              {/* Tips Card */}
              <Card className="health-card">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <BarChart3 className="h-5 w-5 text-primary" />
                    Goal Setting Tips
                  </CardTitle>
                  <CardDescription>Make your goals more achievable</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-start gap-3 p-3 rounded-lg bg-muted/30">
                      <span className="text-lg">🎯</span>
                      <div>
                        <h4 className="font-medium text-sm">Start Small</h4>
                        <p className="text-xs text-muted-foreground">
                          Begin with achievable targets and gradually increase them
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 rounded-lg bg-muted/30">
                      <span className="text-lg">📊</span>
                      <div>
                        <h4 className="font-medium text-sm">Track Daily</h4>
                        <p className="text-xs text-muted-foreground">
                          Regular updates help you stay on track and motivated
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 rounded-lg bg-muted/30">
                      <span className="text-lg">🎉</span>
                      <div>
                        <h4 className="font-medium text-sm">Celebrate Wins</h4>
                        <p className="text-xs text-muted-foreground">
                          Acknowledge your progress, no matter how small
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-3 p-3 rounded-lg bg-muted/30">
                      <span className="text-lg">🔄</span>
                      <div>
                        <h4 className="font-medium text-sm">Be Flexible</h4>
                        <p className="text-xs text-muted-foreground">
                          Adjust goals as needed - progress isn't always linear
                        </p>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="completed" className="space-y-6">
            {completedGoals.length === 0 ? (
              <Card className="health-card">
                <CardContent className="text-center py-12">
                  <Trophy className="h-16 w-16 mx-auto mb-4 text-muted-foreground/30" />
                  <h3 className="font-semibold text-lg mb-2">No completed goals yet</h3>
                  <p className="text-muted-foreground mb-4">
                    Keep working on your active goals to see them here!
                  </p>
                </CardContent>
              </Card>
            ) : (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {completedGoals.map((goal) => (
                  <Card key={goal.id} className="health-card bg-success/5 border-success/20">
                    <CardContent className="pt-6">
                      <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 rounded-lg bg-success/20">
                          <CheckCircle2 className="h-5 w-5 text-success" />
                        </div>
                        <div>
                          <h4 className="font-medium">{goal.goal_name}</h4>
                          <p className="text-xs text-muted-foreground">
                            Completed on {new Date(goal.updated_at).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Target achieved:</span>
                        <span className="font-semibold text-success">
                          {goal.target_value.toLocaleString()} {goal.unit}
                        </span>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
        </Tabs>
      </div>
    </MainLayout>
  );
}
