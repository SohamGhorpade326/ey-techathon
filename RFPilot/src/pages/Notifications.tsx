import { useState } from "react";
import { motion } from "framer-motion";
import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  AlertTriangle,
  Bell,
  CheckCircle2,
  Clock,
  Eye,
  Search,
  Sparkles,
  Trash2,
} from "lucide-react";
import { cn } from "@/lib/utils";

const notifications = [
  {
    id: 1,
    type: "critical",
    title: "Response due in 5 days — high impact risk",
    description: "RFP #1023 - State Electricity Board requires immediate attention. Delays significantly reduce win probability.",
    time: "2 hours ago",
    read: false,
    source: "Master Agent",
  },
  {
    id: 2,
    type: "discovery",
    title: "3 new RFPs discovered on XYZ PSU portal",
    description: "Sales Agent identified matching opportunities in the Power Cable category.",
    time: "4 hours ago",
    read: false,
    source: "Sales Agent",
  },
  {
    id: 3,
    type: "success",
    title: "Spec matching complete for RFP 1023",
    description: "Technical Agent matched 12/12 items with 91% average spec-match. Ready for review.",
    time: "5 hours ago",
    read: false,
    source: "Technical Agent",
  },
  {
    id: 4,
    type: "warning",
    title: "Low spec-match detected",
    description: "Flexible Cable (FLX-EPR-3C-4) has only 68% spec-match. Consider alternative SKU.",
    time: "5 hours ago",
    read: true,
    source: "Technical Agent",
  },
  {
    id: 5,
    type: "info",
    title: "Pricing calculation complete",
    description: "Pricing Agent estimated total bid value at ₹18.7M for RFP #1023.",
    time: "6 hours ago",
    read: true,
    source: "Pricing Agent",
  },
  {
    id: 6,
    type: "discovery",
    title: "High-value RFP detected",
    description: "New tender worth ₹50M+ identified from Central Railways for Signalling Cables.",
    time: "1 day ago",
    read: true,
    source: "Sales Agent",
  },
  {
    id: 7,
    type: "critical",
    title: "Deadline reminder: 10 days remaining",
    description: "RFP #1019 - NTPC Power Station submission deadline approaching.",
    time: "1 day ago",
    read: true,
    source: "Master Agent",
  },
  {
    id: 8,
    type: "success",
    title: "Proposal submitted successfully",
    description: "RFP #1015 bid response has been marked as submitted.",
    time: "2 days ago",
    read: true,
    source: "System",
  },
];

export default function Notifications() {
  const [items, setItems] = useState(notifications);
  const [filter, setFilter] = useState("all");

  const markAsRead = (id: number) => {
    setItems((prev) =>
      prev.map((item) => (item.id === id ? { ...item, read: true } : item))
    );
  };

  const markAllAsRead = () => {
    setItems((prev) => prev.map((item) => ({ ...item, read: true })));
  };

  const deleteNotification = (id: number) => {
    setItems((prev) => prev.filter((item) => item.id !== id));
  };

  const getIcon = (type: string) => {
    switch (type) {
      case "critical":
        return <AlertTriangle className="h-5 w-5 text-destructive" />;
      case "warning":
        return <Clock className="h-5 w-5 text-warning" />;
      case "success":
        return <CheckCircle2 className="h-5 w-5 text-success" />;
      case "discovery":
        return <Search className="h-5 w-5 text-info" />;
      default:
        return <Bell className="h-5 w-5 text-muted-foreground" />;
    }
  };

  const getAgentColor = (source: string) => {
    switch (source) {
      case "Sales Agent":
        return "bg-agent-sales/10 text-agent-sales border-agent-sales/30";
      case "Technical Agent":
        return "bg-agent-technical/10 text-agent-technical border-agent-technical/30";
      case "Pricing Agent":
        return "bg-agent-pricing/10 text-agent-pricing border-agent-pricing/30";
      case "Master Agent":
        return "bg-agent-master/10 text-agent-master border-agent-master/30";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  const filteredItems = items.filter((item) => {
    if (filter === "all") return true;
    if (filter === "unread") return !item.read;
    if (filter === "critical") return item.type === "critical";
    return true;
  });

  const unreadCount = items.filter((item) => !item.read).length;
  const criticalCount = items.filter((item) => item.type === "critical").length;

  return (
    <MainLayout
      title="Notifications"
      breadcrumbs={[
        { name: "Dashboard", href: "/dashboard" },
        { name: "Notifications" },
      ]}
    >
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div className="flex items-center gap-4">
            <Badge variant="outline" className="text-sm">
              {unreadCount} unread
            </Badge>
            <Badge className="bg-destructive/10 text-destructive border border-destructive/30">
              {criticalCount} critical
            </Badge>
          </div>
          <Button variant="outline" onClick={markAllAsRead}>
            <CheckCircle2 className="h-4 w-4 mr-2" />
            Mark all as read
          </Button>
        </motion.div>

        {/* Tabs */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Tabs value={filter} onValueChange={setFilter}>
            <TabsList className="bg-muted/50">
              <TabsTrigger value="all">All</TabsTrigger>
              <TabsTrigger value="unread">Unread</TabsTrigger>
              <TabsTrigger value="critical">Critical</TabsTrigger>
            </TabsList>
          </Tabs>
        </motion.div>

        {/* Notifications List */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-3"
        >
          {filteredItems.map((item, index) => (
            <motion.div
              key={item.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
              className={cn(
                "p-4 rounded-xl border transition-all",
                item.read
                  ? "bg-card border-border"
                  : "bg-primary/5 border-primary/20",
                item.type === "critical" && !item.read && "border-destructive/30 bg-destructive/5"
              )}
            >
              <div className="flex items-start gap-4">
                <div className="p-2 rounded-lg bg-muted/50">{getIcon(item.type)}</div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-3 mb-1">
                    <h3
                      className={cn(
                        "font-medium",
                        !item.read && "text-foreground",
                        item.read && "text-muted-foreground"
                      )}
                    >
                      {item.title}
                    </h3>
                    {!item.read && (
                      <span className="h-2 w-2 rounded-full bg-primary shrink-0" />
                    )}
                  </div>
                  <p className="text-sm text-muted-foreground mb-2">
                    {item.description}
                  </p>
                  <div className="flex items-center gap-3">
                    <Badge
                      variant="outline"
                      className={cn("text-xs", getAgentColor(item.source))}
                    >
                      <Sparkles className="h-3 w-3 mr-1" />
                      {item.source}
                    </Badge>
                    <span className="text-xs text-muted-foreground">{item.time}</span>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {!item.read && (
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => markAsRead(item.id)}
                    >
                      <Eye className="h-4 w-4" />
                    </Button>
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    className="text-muted-foreground hover:text-destructive"
                    onClick={() => deleteNotification(item.id)}
                  >
                    <Trash2 className="h-4 w-4" />
                  </Button>
                </div>
              </div>
            </motion.div>
          ))}

          {filteredItems.length === 0 && (
            <div className="text-center py-12">
              <Bell className="h-12 w-12 text-muted-foreground/30 mx-auto mb-4" />
              <p className="text-muted-foreground">No notifications</p>
            </div>
          )}
        </motion.div>
      </div>
    </MainLayout>
  );
}
