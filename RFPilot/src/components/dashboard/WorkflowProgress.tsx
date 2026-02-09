import { motion } from "framer-motion";
import { Search, Cpu, DollarSign, FileText, Send, Check } from "lucide-react";
import { cn } from "@/lib/utils";

interface WorkflowStep {
  id: string;
  name: string;
  icon: React.ElementType;
  status: "completed" | "active" | "pending";
}

const steps: WorkflowStep[] = [
  { id: "find", name: "Find RFP", icon: Search, status: "completed" },
  { id: "specs", name: "Specs Match", icon: Cpu, status: "completed" },
  { id: "pricing", name: "Pricing", icon: DollarSign, status: "active" },
  { id: "proposal", name: "Proposal", icon: FileText, status: "pending" },
  { id: "submit", name: "Submit", icon: Send, status: "pending" },
];

export function WorkflowProgress() {
  return (
    <motion.div
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="card-elevated p-4"
    >
      <div className="flex items-center justify-between">
        {steps.map((step, index) => (
          <div key={step.id} className="flex items-center">
            {/* Step */}
            <motion.div
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              className="flex items-center gap-2"
            >
              <div
                className={cn(
                  "flex h-10 w-10 items-center justify-center rounded-full transition-all",
                  step.status === "completed" && "bg-success text-success-foreground",
                  step.status === "active" && "bg-primary text-primary-foreground shadow-glow",
                  step.status === "pending" && "bg-muted text-muted-foreground"
                )}
              >
                {step.status === "completed" ? (
                  <Check className="h-5 w-5" />
                ) : (
                  <step.icon className="h-5 w-5" />
                )}
              </div>
              <span
                className={cn(
                  "text-sm font-medium hidden sm:block",
                  step.status === "completed" && "text-success",
                  step.status === "active" && "text-primary",
                  step.status === "pending" && "text-muted-foreground"
                )}
              >
                {step.name}
              </span>
            </motion.div>

            {/* Connector */}
            {index < steps.length - 1 && (
              <div className="w-12 lg:w-20 mx-2">
                <motion.div
                  initial={{ scaleX: 0 }}
                  animate={{ scaleX: 1 }}
                  transition={{ duration: 0.4, delay: index * 0.15 }}
                  className={cn(
                    "h-0.5 origin-left",
                    step.status === "completed" ? "bg-success" : "bg-border"
                  )}
                />
              </div>
            )}
          </div>
        ))}
      </div>
    </motion.div>
  );
}
