import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { useHealthMonitor } from '@/hooks/useHealthMonitor';
import { Lightbulb, AlertTriangle, CheckCircle, Info } from 'lucide-react';
import { cn } from '@/lib/utils';

export function HealthTipsCard() {
  const { healthTips, healthAlerts, isLoading } = useHealthMonitor();

  if (isLoading) {
    return (
      <Card className="shadow-card border-border/40">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Lightbulb className="h-5 w-5 text-warning" />
            Health Tips
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="animate-pulse space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-20 bg-muted rounded-lg" />
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="shadow-card border-border/40 hover:shadow-card-hover transition-all duration-300">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Lightbulb className="h-5 w-5 text-warning" />
          Personalized Health Tips
        </CardTitle>
        <CardDescription>Based on your lifestyle data</CardDescription>
      </CardHeader>
      <CardContent>
        {/* Health Alerts */}
        {healthAlerts.length > 0 && (
          <div className="mb-4 space-y-2">
            {healthAlerts.map((alert) => (
              <div
                key={alert.id}
                className={cn(
                  'p-3 rounded-lg flex items-start gap-3',
                  alert.severity === 'warning' && 'bg-warning/10 border border-warning/30',
                  alert.severity === 'success' && 'bg-success/10 border border-success/30',
                  alert.severity === 'info' && 'bg-info/10 border border-info/30'
                )}
              >
                {alert.severity === 'warning' && (
                  <AlertTriangle className="h-5 w-5 text-warning flex-shrink-0 mt-0.5" />
                )}
                {alert.severity === 'success' && (
                  <CheckCircle className="h-5 w-5 text-success flex-shrink-0 mt-0.5" />
                )}
                {alert.severity === 'info' && (
                  <Info className="h-5 w-5 text-info flex-shrink-0 mt-0.5" />
                )}
                <div className="flex-1">
                  <p className="text-sm font-medium">{alert.message}</p>
                  {alert.action && (
                    <p className="text-xs text-muted-foreground mt-1">
                      Recommended: {alert.action}
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Health Tips */}
        <ScrollArea className="h-[300px] pr-4">
          <div className="space-y-3">
            {healthTips.slice(0, 5).map((tip) => (
              <div
                key={tip.id}
                className="p-4 bg-muted/30 rounded-xl hover:bg-muted/50 transition-colors"
              >
                <div className="flex items-start gap-3">
                  <span className="text-2xl">{tip.icon}</span>
                  <div className="flex-1">
                    <div className="flex items-center gap-2 mb-1">
                      <h4 className="font-semibold text-sm">{tip.title}</h4>
                      <Badge
                        variant="outline"
                        className={cn(
                          'text-[10px] px-1.5 py-0',
                          tip.priority === 'high' && 'border-destructive/50 text-destructive',
                          tip.priority === 'medium' && 'border-warning/50 text-warning',
                          tip.priority === 'low' && 'border-success/50 text-success'
                        )}
                      >
                        {tip.priority}
                      </Badge>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed">
                      {tip.description}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
