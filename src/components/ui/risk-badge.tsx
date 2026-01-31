import { AlertTriangle, CheckCircle, XCircle } from "lucide-react";

interface RiskBadgeProps {
  riskLevel: "low" | "medium" | "high";
}

const RISK_CONFIG = {
  low: {
    label: "Low",
    icon: CheckCircle,
    className: "bg-green-500/10 text-green-600 border-green-500/20",
  },
  medium: {
    label: "Medium",
    icon: AlertTriangle,
    className: "bg-yellow-500/10 text-yellow-600 border-yellow-500/20",
  },
  high: {
    label: "High",
    icon: XCircle,
    className: "bg-red-500/10 text-red-600 border-red-500/20",
  },
} as const;

export function RiskBadge({ riskLevel }: RiskBadgeProps) {
  const config = RISK_CONFIG[riskLevel];

  // 🛡️ SAFETY GUARD (prevents white screen forever)
  if (!config) {
    return (
      <span className="px-3 py-1 text-sm rounded-full border">
        Unknown
      </span>
    );
  }

  const Icon = config.icon;

  return (
    <span
      className={`inline-flex items-center gap-2 px-3 py-1 text-sm font-medium rounded-full border ${config.className}`}
    >
      <Icon className="h-4 w-4" />
      {config.label}
    </span>
  );
}
