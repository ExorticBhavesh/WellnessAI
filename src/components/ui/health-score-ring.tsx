import { cn } from '@/lib/utils';

interface HealthScoreRingProps {
  score: number;
  maxScore?: number;
  size?: 'sm' | 'md' | 'lg' | 'xl';
  showLabel?: boolean;
  label?: string;
  className?: string;
}

const sizeClasses = {
  sm: { container: 'h-16 w-16', stroke: 4, text: 'text-lg', label: 'text-xs' },
  md: { container: 'h-24 w-24', stroke: 6, text: 'text-2xl', label: 'text-xs' },
  lg: { container: 'h-32 w-32', stroke: 8, text: 'text-3xl', label: 'text-sm' },
  xl: { container: 'h-44 w-44', stroke: 10, text: 'text-4xl', label: 'text-sm' },
};

export function HealthScoreRing({
  score,
  maxScore = 100,
  size = 'md',
  showLabel = true,
  label = 'Health Score',
  className,
}: HealthScoreRingProps) {
  const percentage = Math.min(100, Math.max(0, (score / maxScore) * 100));
  const { container, stroke, text, label: labelSize } = sizeClasses[size];
  
  // Calculate circle properties
  const radius = 50 - stroke / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (percentage / 100) * circumference;

  // Determine color based on score
  const getColor = () => {
    if (percentage >= 70) return 'hsl(var(--success))';
    if (percentage >= 40) return 'hsl(var(--warning))';
    return 'hsl(var(--destructive))';
  };

  return (
    <div className={cn('flex flex-col items-center gap-2', className)}>
      <div className={cn('relative', container)}>
        <svg className="h-full w-full -rotate-90" viewBox="0 0 100 100">
          {/* Background circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke="hsl(var(--muted))"
            strokeWidth={stroke}
          />
          {/* Progress circle */}
          <circle
            cx="50"
            cy="50"
            r={radius}
            fill="none"
            stroke={getColor()}
            strokeWidth={stroke}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-1000 ease-out"
          />
        </svg>
        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className={cn('font-display font-bold', text)}>{Math.round(score)}</span>
        </div>
      </div>
      {showLabel && (
        <span className={cn('font-medium text-muted-foreground', labelSize)}>{label}</span>
      )}
    </div>
  );
}