import { motion } from "framer-motion";
import { AlertTriangle, Clock, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { Link } from "react-router-dom";

interface UrgentRFP {
  id: string;
  title: string;
  buyer: string;
  daysRemaining: number;
  value: string;
  priority: "critical" | "high" | "medium";
}

const mockUrgentRFPs: UrgentRFP[] = [
  {
    id: "1023",
    title: "132kV XLPE Power Cables Supply",
    buyer: "PGCIL",
    daysRemaining: 3,
    value: "₹45.2 Cr",
    priority: "critical",
  },
  {
    id: "1024",
    title: "LT Aerial Bunched Cables Package",
    buyer: "MSEDCL",
    daysRemaining: 7,
    value: "₹12.8 Cr",
    priority: "high",
  },
  {
    id: "1025",
    title: "Control & Instrumentation Cables",
    buyer: "NTPC",
    daysRemaining: 10,
    value: "₹8.5 Cr",
    priority: "medium",
  },
];

export function UrgentAlertsPanel() {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.2 }}
      className="card-elevated p-6"
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <AlertTriangle className="h-5 w-5 text-warning" />
          <h3 className="text-lg font-semibold text-foreground">Urgent RFPs</h3>
        </div>
        <Link to="/discovery">
          <Button variant="ghost" size="sm" className="gap-1">
            View All <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>

      <div className="space-y-3">
        {mockUrgentRFPs.map((rfp, index) => (
          <motion.div
            key={rfp.id}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.3 + index * 0.1 }}
            className={cn(
              "flex items-center gap-4 p-4 rounded-lg border transition-all hover:shadow-md cursor-pointer",
              rfp.priority === "critical" && "border-destructive/30 bg-destructive/5",
              rfp.priority === "high" && "border-warning/30 bg-warning/5",
              rfp.priority === "medium" && "border-border bg-card"
            )}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-muted-foreground">
                  #{rfp.id}
                </span>
                <span
                  className={cn(
                    "badge-status",
                    rfp.priority === "critical" && "badge-urgent",
                    rfp.priority === "high" && "bg-warning/10 text-warning",
                    rfp.priority === "medium" && "bg-muted text-muted-foreground"
                  )}
                >
                  {rfp.priority}
                </span>
              </div>
              <p className="font-medium text-foreground truncate mt-1">
                {rfp.title}
              </p>
              <p className="text-sm text-muted-foreground">{rfp.buyer}</p>
            </div>

            <div className="text-right shrink-0">
              <div className="flex items-center gap-1 text-foreground font-semibold">
                <Clock className="h-4 w-4 text-muted-foreground" />
                <span
                  className={cn(
                    rfp.daysRemaining <= 5 && "text-destructive",
                    rfp.daysRemaining <= 10 && rfp.daysRemaining > 5 && "text-warning"
                  )}
                >
                  {rfp.daysRemaining} days
                </span>
              </div>
              <p className="text-sm text-muted-foreground">{rfp.value}</p>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-4 p-3 rounded-lg bg-info/10 border border-info/20">
        <p className="text-xs text-info">
          <strong>Insight:</strong> 90% of wins correlate with timely action on RFPs.
          Delays significantly reduce win probability.
        </p>
      </div>
    </motion.div>
  );
}
