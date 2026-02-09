import { motion } from "framer-motion";
import { Search, Cpu, DollarSign, Sparkles, CheckCircle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface AgentActivity {
  id: string;
  agent: "sales" | "technical" | "pricing" | "master";
  message: string;
  status: "completed" | "in-progress" | "pending";
  timestamp: string;
}

const agentConfig = {
  sales: {
    name: "Sales Agent",
    icon: Search,
    color: "text-agent-sales",
    bgColor: "bg-agent-sales/10",
  },
  technical: {
    name: "Technical Agent",
    icon: Cpu,
    color: "text-agent-technical",
    bgColor: "bg-agent-technical/10",
  },
  pricing: {
    name: "Pricing Agent",
    icon: DollarSign,
    color: "text-agent-pricing",
    bgColor: "bg-agent-pricing/10",
  },
  master: {
    name: "Master Agent",
    icon: Sparkles,
    color: "text-agent-master",
    bgColor: "bg-agent-master/10",
  },
};

const mockActivities: AgentActivity[] = [
  {
    id: "1",
    agent: "sales",
    message: "Monitoring 12 PSU portals for new RFPs...",
    status: "in-progress",
    timestamp: "Just now",
  },
  {
    id: "2",
    agent: "sales",
    message: "Discovered 3 new RFPs from PGCIL portal",
    status: "completed",
    timestamp: "5 min ago",
  },
  {
    id: "3",
    agent: "technical",
    message: "Completed spec matching for RFP #1023",
    status: "completed",
    timestamp: "12 min ago",
  },
  {
    id: "4",
    agent: "pricing",
    message: "Estimating costs for 15 SKU items...",
    status: "in-progress",
    timestamp: "15 min ago",
  },
  {
    id: "5",
    agent: "master",
    message: "Ready to consolidate proposal for RFP #1021",
    status: "pending",
    timestamp: "20 min ago",
  },
];

export function AgentActivityPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.4, delay: 0.3 }}
      className="card-elevated p-6 h-full"
    >
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-foreground">Agent Activity</h3>
        <span className="text-xs text-muted-foreground">Live updates</span>
      </div>

      <div className="space-y-4">
        {mockActivities.map((activity, index) => {
          const config = agentConfig[activity.agent];
          const Icon = config.icon;

          return (
            <motion.div
              key={activity.id}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.3, delay: 0.4 + index * 0.1 }}
              className="flex items-start gap-3"
            >
              <div className={cn("rounded-lg p-2", config.bgColor)}>
                <Icon className={cn("h-4 w-4", config.color)} />
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <p className={cn("text-sm font-medium", config.color)}>
                    {config.name}
                  </p>
                  {activity.status === "in-progress" && (
                    <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
                  )}
                  {activity.status === "completed" && (
                    <CheckCircle className="h-3 w-3 text-success" />
                  )}
                </div>
                <p className="text-sm text-muted-foreground truncate">
                  {activity.message}
                </p>
                <p className="text-xs text-muted-foreground/60 mt-0.5">
                  {activity.timestamp}
                </p>
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
