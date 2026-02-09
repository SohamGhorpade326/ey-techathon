import { motion } from "framer-motion";
import { TrendingUp } from "lucide-react";

export function WinCorrelationChart() {
  const data = [
    { label: "Timely Response", value: 90, color: "bg-success" },
    { label: "Technical Alignment", value: 60, color: "bg-info" },
    { label: "Price Competitiveness", value: 75, color: "bg-primary" },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, delay: 0.35 }}
      className="card-elevated p-6"
    >
      <div className="flex items-center gap-2 mb-6">
        <TrendingUp className="h-5 w-5 text-primary" />
        <h3 className="text-lg font-semibold text-foreground">Win Correlation Factors</h3>
      </div>

      <div className="space-y-5">
        {data.map((item, index) => (
          <motion.div
            key={item.label}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.3, delay: 0.4 + index * 0.1 }}
          >
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-foreground">{item.label}</span>
              <span className="text-sm font-semibold text-foreground">{item.value}%</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${item.value}%` }}
                transition={{ duration: 0.6, delay: 0.5 + index * 0.1, ease: "easeOut" }}
                className={`h-full rounded-full ${item.color}`}
              />
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-6 pt-4 border-t border-border">
        <p className="text-xs text-muted-foreground">
          Based on analysis of 150+ RFP submissions over the past 12 months. Timely responses
          (within first 50% of deadline) show highest win correlation.
        </p>
      </div>
    </motion.div>
  );
}
