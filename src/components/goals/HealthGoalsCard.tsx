import { useState } from 'react';
import { Target, Plus, Footprints, Moon, Droplets, Dumbbell, Scale, Trash2, CheckCircle2 } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Progress } from '@/components/ui/progress';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useHealthGoals, GOAL_PRESETS, HealthGoal } from '@/hooks/useHealthGoals';
import { cn } from '@/lib/utils';

const iconMap = {
  Footprints,
  Moon,
  Droplets,
  Dumbbell,
  Scale,
  Target,
};

export function HealthGoalsCard() {
  const { activeGoals, isLoading, addGoal, updateGoal, deleteGoal, getProgress, isAdding } = useHealthGoals();
  const [isOpen, setIsOpen] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<string>('');
  const [customName, setCustomName] = useState('');
  const [targetValue, setTargetValue] = useState('');
  const [unit, setUnit] = useState('');

  const handlePresetChange = (value: string) => {
    setSelectedPreset(value);
    const preset = GOAL_PRESETS.find(p => p.type === value);
    if (preset) {
      setCustomName(preset.name);
      setTargetValue(preset.target.toString());
      setUnit(preset.unit);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedPreset || !targetValue) return;

    addGoal({
      goal_type: selectedPreset,
      goal_name: customName || GOAL_PRESETS.find(p => p.type === selectedPreset)?.name || 'Custom Goal',
      target_value: parseFloat(targetValue),
      unit: unit || 'units',
    });
    
    setIsOpen(false);
    setSelectedPreset('');
    setCustomName('');
    setTargetValue('');
    setUnit('');
  };

  const handleUpdateProgress = (goal: HealthGoal, increment: number) => {
    const newValue = Math.max(0, goal.current_value + increment);
    updateGoal({
      id: goal.id,
      current_value: newValue,
      is_active: newValue < goal.target_value,
    });
  };

  const getGoalIcon = (goalType: string) => {
    const preset = GOAL_PRESETS.find(p => p.type === goalType);
    const IconComponent = preset ? iconMap[preset.icon as keyof typeof iconMap] : Target;
    return IconComponent;
  };

  const getProgressColor = (progress: number) => {
    if (progress >= 100) return 'bg-success';
    if (progress >= 75) return 'bg-primary';
    if (progress >= 50) return 'bg-info';
    if (progress >= 25) return 'bg-warning';
    return 'bg-muted-foreground';
  };

  if (isLoading) {
    return (
      <Card className="health-card">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            Health Goals
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-4">
            {[1, 2].map(i => (
              <div key={i} className="h-20 bg-muted rounded-xl" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="health-card">
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <div>
          <CardTitle className="flex items-center gap-2">
            <Target className="h-5 w-5 text-primary" />
            Health Goals
          </CardTitle>
          <CardDescription>Track your wellness objectives</CardDescription>
        </div>
        <Dialog open={isOpen} onOpenChange={setIsOpen}>
          <DialogTrigger asChild>
            <Button size="sm" variant="outline" className="gap-1">
              <Plus className="h-4 w-4" />
              Add Goal
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Create Health Goal</DialogTitle>
              <DialogDescription>Set a new wellness objective to track</DialogDescription>
            </DialogHeader>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label>Goal Type</Label>
                <Select value={selectedPreset} onValueChange={handlePresetChange}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select a goal type" />
                  </SelectTrigger>
                  <SelectContent>
                    {GOAL_PRESETS.map(preset => {
                      const Icon = iconMap[preset.icon as keyof typeof iconMap];
                      return (
                        <SelectItem key={preset.type} value={preset.type}>
                          <div className="flex items-center gap-2">
                            <Icon className="h-4 w-4" />
                            {preset.name}
                          </div>
                        </SelectItem>
                      );
                    })}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Goal Name</Label>
                <Input
                  value={customName}
                  onChange={e => setCustomName(e.target.value)}
                  placeholder="e.g., Daily Steps Goal"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label>Target Value</Label>
                  <Input
                    type="number"
                    value={targetValue}
                    onChange={e => setTargetValue(e.target.value)}
                    placeholder="e.g., 10000"
                  />
                </div>
                <div className="space-y-2">
                  <Label>Unit</Label>
                  <Input
                    value={unit}
                    onChange={e => setUnit(e.target.value)}
                    placeholder="e.g., steps"
                  />
                </div>
              </div>

              <Button type="submit" className="w-full" disabled={isAdding || !selectedPreset || !targetValue}>
                {isAdding ? 'Creating...' : 'Create Goal'}
              </Button>
            </form>
          </DialogContent>
        </Dialog>
      </CardHeader>

      <CardContent className="space-y-4">
        {activeGoals.length === 0 ? (
          <div className="text-center py-8 text-muted-foreground">
            <Target className="h-12 w-12 mx-auto mb-3 opacity-50" />
            <p className="font-medium">No active goals</p>
            <p className="text-sm">Create a goal to start tracking your wellness</p>
          </div>
        ) : (
          activeGoals.map(goal => {
            const Icon = getGoalIcon(goal.goal_type);
            const progress = getProgress(goal);
            const isComplete = progress >= 100;

            return (
              <div
                key={goal.id}
                className={cn(
                  "p-4 rounded-xl border transition-all",
                  isComplete 
                    ? "bg-success/10 border-success/30" 
                    : "bg-muted/30 border-border/50 hover:border-primary/30"
                )}
              >
                <div className="flex items-start justify-between mb-3">
                  <div className="flex items-center gap-3">
                    <div className={cn(
                      "p-2 rounded-lg",
                      isComplete ? "bg-success/20" : "bg-primary/10"
                    )}>
                      <Icon className={cn(
                        "h-5 w-5",
                        isComplete ? "text-success" : "text-primary"
                      )} />
                    </div>
                    <div>
                      <h4 className="font-medium flex items-center gap-2">
                        {goal.goal_name}
                        {isComplete && <CheckCircle2 className="h-4 w-4 text-success" />}
                      </h4>
                      <p className="text-sm text-muted-foreground">
                        {goal.current_value.toLocaleString()} / {goal.target_value.toLocaleString()} {goal.unit}
                      </p>
                    </div>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-muted-foreground hover:text-destructive"
                    onClick={() => deleteGoal(goal.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>

                <div className="space-y-2">
                  <Progress 
                    value={progress} 
                    className={cn("h-2", isComplete && "[&>div]:bg-success")}
                  />
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-muted-foreground">
                      {progress.toFixed(0)}% complete
                    </span>
                    <div className="flex gap-1">
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => handleUpdateProgress(goal, -getIncrement(goal.goal_type))}
                        disabled={goal.current_value <= 0}
                      >
                        -
                      </Button>
                      <Button
                        variant="outline"
                        size="sm"
                        className="h-7 px-2 text-xs"
                        onClick={() => handleUpdateProgress(goal, getIncrement(goal.goal_type))}
                      >
                        +
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </CardContent>
    </Card>
  );
}

function getIncrement(goalType: string): number {
  switch (goalType) {
    case 'steps': return 1000;
    case 'sleep': return 0.5;
    case 'water': return 1;
    case 'exercise': return 5;
    case 'weight': return 0.1;
    default: return 1;
  }
}
