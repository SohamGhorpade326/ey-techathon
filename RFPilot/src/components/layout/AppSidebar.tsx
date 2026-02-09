import { useLocation, Link } from "react-router-dom";
import { motion } from "framer-motion";
import {
  LayoutDashboard,
  Search,
  Brain,
  DollarSign,
  FileText,
  Bell,
  Database,
  Settings,
  LogOut,
  Sparkles,
  Zap,
  Package,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navigation = [
  { name: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
  { name: "RFP Discovery", href: "/discovery", icon: Search },
  { name: "Strategic Orchestrator", href: "/master-agent", icon: Zap },
  { name: "AI Analysis Workspace", href: "/analysis", icon: Brain },
 
  { name: "Pricing & Summary", href: "/pricing", icon: DollarSign },
  { name: "Proposal Builder", href: "/proposal", icon: FileText },
   { name: "SKU Gap Intelligence", href: "/sku-gap-analytics", icon: Package },
  { name: "Notifications", href: "/notifications", icon: Bell, badge: 3 },
  { name: "Product Repository", href: "/repository", icon: Database },
  { name: "Settings", href: "/settings", icon: Settings },
];

export function AppSidebar() {
  const location = useLocation();

  return (
    <motion.aside
      initial={{ x: -280 }}
      animate={{ x: 0 }}
      transition={{ duration: 0.3, ease: "easeOut" }}
      className="fixed left-0 top-0 z-40 h-screen w-[280px] bg-sidebar border-r border-sidebar-border"
      style={{ background: "var(--gradient-sidebar)" }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-6 border-b border-sidebar-border">
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-sidebar-primary">
          <Sparkles className="h-5 w-5 text-sidebar-primary-foreground" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-sidebar-foreground">Agentic RFP</h1>
          <p className="text-xs text-sidebar-muted">Response Platform</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-1 overflow-y-auto">
        {navigation.map((item) => {
          const isActive = location.pathname === item.href;
          return (
            <Link
              key={item.name}
              to={item.href}
              className={cn(
                "sidebar-link group relative",
                isActive && "active"
              )}
            >
              {isActive && (
                <motion.div
                  layoutId="sidebar-active"
                  className="absolute inset-0 bg-sidebar-primary rounded-lg"
                  transition={{ type: "spring", bounce: 0.2, duration: 0.4 }}
                />
              )}
              <span className="relative z-10 flex items-center gap-3">
                <item.icon className="h-5 w-5" />
                <span>{item.name}</span>
                {item.badge && (
                  <span className="ml-auto flex h-5 min-w-5 items-center justify-center rounded-full bg-destructive px-1.5 text-xs font-medium text-destructive-foreground">
                    {item.badge}
                  </span>
                )}
              </span>
            </Link>
          );
        })}
      </nav>

      {/* Agent Status */}
      <div className="px-4 py-4 border-t border-sidebar-border">
        <div className="rounded-lg bg-sidebar-accent p-4">
          <p className="text-xs font-medium text-sidebar-muted mb-3 uppercase tracking-wider">
            Agent Status
          </p>
          <div className="space-y-2">
            <AgentStatus name="Sales Agent" status="active" color="bg-agent-sales" />
            <AgentStatus name="Technical Agent" status="idle" color="bg-agent-technical" />
            <AgentStatus name="Pricing Agent" status="idle" color="bg-agent-pricing" />
            <AgentStatus name="Master Agent" status="ready" color="bg-agent-master" />
          </div>
        </div>
      </div>

      {/* User */}
      <div className="px-4 py-4 border-t border-sidebar-border">
        <button className="sidebar-link w-full text-sidebar-muted hover:text-sidebar-foreground">
          <LogOut className="h-5 w-5" />
          <span>Sign Out</span>
        </button>
      </div>
    </motion.aside>
  );
}

function AgentStatus({
  name,
  status,
  color,
}: {
  name: string;
  status: "active" | "idle" | "ready";
  color: string;
}) {
  return (
    <div className="flex items-center gap-2">
      <div className="relative">
        <div className={cn("h-2 w-2 rounded-full", color)} />
        {status === "active" && (
          <div className={cn("absolute inset-0 h-2 w-2 rounded-full animate-ping", color, "opacity-50")} />
        )}
      </div>
      <span className="text-xs text-sidebar-foreground/70">{name}</span>
      <span
        className={cn(
          "ml-auto text-xs",
          status === "active" && "text-agent-sales",
          status === "idle" && "text-sidebar-muted",
          status === "ready" && "text-agent-master"
        )}
      >
        {status}
      </span>
    </div>
  );
}
