import { motion } from "framer-motion";
import { FileText, Clock, Target, CheckCircle, ArrowRight } from "lucide-react";
import { Link } from "react-router-dom";
import { MainLayout } from "@/components/layout/MainLayout";
import { KPICard } from "@/components/dashboard/KPICard";
import { AgentActivityPanel } from "@/components/dashboard/AgentActivityPanel";
import { UrgentAlertsPanel } from "@/components/dashboard/UrgentAlertsPanel";
import { WorkflowProgress } from "@/components/dashboard/WorkflowProgress";
import { WinCorrelationChart } from "@/components/dashboard/WinCorrelationChart";
import { Button } from "@/components/ui/button";

export default function Dashboard() {
  return (
    <MainLayout title="Dashboard">
      {/* Workflow Progress Bar */}
      <WorkflowProgress />

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mt-6">
        <KPICard
          title="Qualified RFPs in Pipeline"
          value={24}
          subtitle="Active opportunities"
          icon={FileText}
          trend={{ value: 12, isPositive: true }}
          accentColor="primary"
          delay={0.1}
        />
        <KPICard
          title="RFPs Due in 90 Days"
          value={18}
          subtitle="Critical deadline window"
          icon={Clock}
          trend={{ value: 8, isPositive: true }}
          accentColor="warning"
          delay={0.15}
        />
        <KPICard
          title="Avg. Spec-Match %"
          value="87%"
          subtitle="Technical alignment score"
          icon={Target}
          trend={{ value: 5, isPositive: true }}
          accentColor="success"
          delay={0.2}
        />
        <KPICard
          title="Submissions This Month"
          value={8}
          subtitle="5 wins (62.5% rate)"
          icon={CheckCircle}
          trend={{ value: 15, isPositive: true }}
          accentColor="info"
          delay={0.25}
        />
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
        {/* Left Column - Urgent Alerts & Win Correlation */}
        <div className="lg:col-span-2 space-y-6">
          <UrgentAlertsPanel />
          <WinCorrelationChart />
        </div>

        {/* Right Column - Agent Activity */}
        <div className="lg:col-span-1">
          <AgentActivityPanel />
        </div>
      </div>

      {/* CTA Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4, delay: 0.5 }}
        className="mt-6 card-elevated p-6 flex flex-col sm:flex-row items-center justify-between gap-4"
      >
        <div>
          <h3 className="text-lg font-semibold text-foreground">
            3 New RFPs Discovered Today
          </h3>
          <p className="text-sm text-muted-foreground">
            Sales Agent found matching opportunities from PGCIL, MSEDCL, and BHEL portals.
          </p>
        </div>
        <Link to="/discovery">
          <Button variant="default" size="lg" className="gap-2 shrink-0">
            Review New RFPs <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </motion.div>
    </MainLayout>
  );
}
