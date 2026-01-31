import { format, formatDistanceToNow } from 'date-fns';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { RiskBadge } from '@/components/ui/risk-badge';
import { useSymptomHistory, SymptomCheck } from '@/hooks/useSymptomHistory';
import { SYMPTOMS_LIST } from '@/lib/constants';
import { History, Stethoscope, ChevronRight, Loader2, AlertCircle } from 'lucide-react';
import { cn } from '@/lib/utils';
import { useState } from 'react';

export function SymptomHistoryTimeline() {
  const { history, isLoading } = useSymptomHistory();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  const getSymptomLabel = (id: string) => {
    const symptom = SYMPTOMS_LIST.find(s => s.id === id);
    return symptom?.label || id;
  };

  const getRiskColor = (level: string | null) => {
    switch (level) {
      case 'high': return 'border-l-destructive';
      case 'medium': return 'border-l-warning';
      case 'low': return 'border-l-success';
      default: return 'border-l-muted';
    }
  };

  if (isLoading) {
    return (
      <Card className="shadow-card border-border/50">
        <CardContent className="flex items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        </CardContent>
      </Card>
    );
  }

  if (history.length === 0) {
    return (
      <Card className="shadow-card border-border/50">
        <CardContent className="flex flex-col items-center justify-center py-12 text-center">
          <History className="h-12 w-12 text-muted-foreground/30 mb-4" />
          <p className="text-muted-foreground">No symptom checks yet</p>
          <p className="text-sm text-muted-foreground/70">
            Your health check history will appear here
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="shadow-card border-border/50">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <History className="h-5 w-5" />
          Symptom History
        </CardTitle>
        <CardDescription>
          Your past health checks and AI recommendations
        </CardDescription>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[500px] pr-4">
          <div className="relative">
            {/* Timeline line */}
            <div className="absolute left-4 top-0 bottom-0 w-0.5 bg-border" />

            <div className="space-y-4">
              {history.map((check, index) => (
                <TimelineItem
                  key={check.id}
                  check={check}
                  isExpanded={expandedId === check.id}
                  onToggle={() => setExpandedId(expandedId === check.id ? null : check.id)}
                  getSymptomLabel={getSymptomLabel}
                  getRiskColor={getRiskColor}
                  isFirst={index === 0}
                />
              ))}
            </div>
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

interface TimelineItemProps {
  check: SymptomCheck;
  isExpanded: boolean;
  onToggle: () => void;
  getSymptomLabel: (id: string) => string;
  getRiskColor: (level: string | null) => string;
  isFirst: boolean;
}

function TimelineItem({ 
  check, 
  isExpanded, 
  onToggle, 
  getSymptomLabel, 
  getRiskColor,
  isFirst 
}: TimelineItemProps) {
  return (
    <div className="relative pl-10">
      {/* Timeline dot */}
      <div className={cn(
        'absolute left-2.5 w-3 h-3 rounded-full border-2 bg-background',
        check.risk_level === 'high' ? 'border-destructive' :
        check.risk_level === 'medium' ? 'border-warning' :
        'border-success'
      )} />

      <div
        className={cn(
          'rounded-lg border-l-4 border bg-card p-4 cursor-pointer transition-all hover:shadow-md',
          getRiskColor(check.risk_level),
          isExpanded && 'shadow-md'
        )}
        onClick={onToggle}
      >
        {/* Header */}
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <Stethoscope className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium">
                {check.symptoms.length} symptom{check.symptoms.length !== 1 ? 's' : ''} checked
              </span>
              {isFirst && (
                <Badge variant="secondary" className="text-xs">Latest</Badge>
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {formatDistanceToNow(new Date(check.created_at), { addSuffix: true })}
              {' • '}
              {format(new Date(check.created_at), 'MMM d, yyyy h:mm a')}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {check.risk_level && <RiskBadge level={check.risk_level} />}
            <ChevronRight className={cn(
              'h-4 w-4 text-muted-foreground transition-transform',
              isExpanded && 'rotate-90'
            )} />
          </div>
        </div>

        {/* Symptoms Tags */}
        <div className="flex flex-wrap gap-1 mt-3">
          {check.symptoms.slice(0, isExpanded ? undefined : 3).map((symptom) => (
            <Badge key={symptom} variant="outline" className="text-xs">
              {getSymptomLabel(symptom)}
            </Badge>
          ))}
          {!isExpanded && check.symptoms.length > 3 && (
            <Badge variant="outline" className="text-xs text-muted-foreground">
              +{check.symptoms.length - 3} more
            </Badge>
          )}
        </div>

        {/* Expanded Content */}
        {isExpanded && (
          <div className="mt-4 pt-4 border-t space-y-4 animate-fade-in">
            {check.ai_analysis && (
              <div>
                <p className="text-sm font-medium mb-2 flex items-center gap-2">
                  <AlertCircle className="h-4 w-4" />
                  AI Analysis
                </p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {check.ai_analysis}
                </p>
              </div>
            )}

            {check.recommendations && check.recommendations.length > 0 && (
              <div>
                <p className="text-sm font-medium mb-2">Recommendations</p>
                <ul className="space-y-2">
                  {check.recommendations.map((rec, i) => (
                    <li key={i} className="text-sm text-muted-foreground flex gap-2">
                      <span className="text-primary">•</span>
                      <span>{rec}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
