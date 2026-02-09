import { useState } from "react";
import { motion } from "framer-motion";
import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Separator } from "@/components/ui/separator";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Bell,
  Building2,
  Lock,
  Save,
  Settings as SettingsIcon,
  Sparkles,
  User,
} from "lucide-react";

export default function Settings() {
  const [notifications, setNotifications] = useState({
    email: true,
    deadlineAlerts: true,
    agentUpdates: true,
    weeklyDigest: false,
  });

  return (
    <MainLayout
      title="Settings"
      breadcrumbs={[
        { name: "Dashboard", href: "/dashboard" },
        { name: "Settings" },
      ]}
    >
      <div className="max-w-4xl">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <Tabs defaultValue="profile" className="space-y-6">
            <TabsList className="bg-muted/50">
              <TabsTrigger value="profile" className="gap-2">
                <User className="h-4 w-4" />
                Profile
              </TabsTrigger>
              <TabsTrigger value="company" className="gap-2">
                <Building2 className="h-4 w-4" />
                Company
              </TabsTrigger>
              <TabsTrigger value="notifications" className="gap-2">
                <Bell className="h-4 w-4" />
                Notifications
              </TabsTrigger>
              <TabsTrigger value="agents" className="gap-2">
                <Sparkles className="h-4 w-4" />
                AI Agents
              </TabsTrigger>
              <TabsTrigger value="security" className="gap-2">
                <Lock className="h-4 w-4" />
                Security
              </TabsTrigger>
            </TabsList>

            {/* Profile Tab */}
            <TabsContent value="profile">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-card rounded-xl border border-border p-6 space-y-6"
              >
                <div>
                  <h3 className="font-semibold mb-1">Personal Information</h3>
                  <p className="text-sm text-muted-foreground">
                    Update your personal details
                  </p>
                </div>
                <Separator />
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label>Full Name</Label>
                    <Input defaultValue="Rajesh Kumar" />
                  </div>
                  <div className="space-y-2">
                    <Label>Job Title</Label>
                    <Input defaultValue="Sales Lead" />
                  </div>
                  <div className="space-y-2">
                    <Label>Email Address</Label>
                    <Input defaultValue="rajesh.kumar@company.com" type="email" />
                  </div>
                  <div className="space-y-2">
                    <Label>Phone Number</Label>
                    <Input defaultValue="+91 98765 43210" />
                  </div>
                </div>
                <div className="flex justify-end">
                  <Button>
                    <Save className="h-4 w-4 mr-2" />
                    Save Changes
                  </Button>
                </div>
              </motion.div>
            </TabsContent>

            {/* Company Tab */}
            <TabsContent value="company">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-card rounded-xl border border-border p-6 space-y-6"
              >
                <div>
                  <h3 className="font-semibold mb-1">Company Details</h3>
                  <p className="text-sm text-muted-foreground">
                    Used in proposal generation and bid responses
                  </p>
                </div>
                <Separator />
                <div className="grid grid-cols-2 gap-6">
                  <div className="space-y-2">
                    <Label>Company Name</Label>
                    <Input defaultValue="Premier Cables India Pvt. Ltd." />
                  </div>
                  <div className="space-y-2">
                    <Label>Registration Number</Label>
                    <Input defaultValue="U31200MH2010PTC123456" />
                  </div>
                  <div className="space-y-2">
                    <Label>GST Number</Label>
                    <Input defaultValue="27AABCP1234R1ZX" />
                  </div>
                  <div className="space-y-2">
                    <Label>PAN Number</Label>
                    <Input defaultValue="AABCP1234R" />
                  </div>
                  <div className="col-span-2 space-y-2">
                    <Label>Registered Address</Label>
                    <Input defaultValue="Plot No. 45, MIDC Industrial Area, Thane, Maharashtra 400601" />
                  </div>
                </div>
                <div className="flex justify-end">
                  <Button>
                    <Save className="h-4 w-4 mr-2" />
                    Save Changes
                  </Button>
                </div>
              </motion.div>
            </TabsContent>

            {/* Notifications Tab */}
            <TabsContent value="notifications">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-card rounded-xl border border-border p-6 space-y-6"
              >
                <div>
                  <h3 className="font-semibold mb-1">Notification Preferences</h3>
                  <p className="text-sm text-muted-foreground">
                    Configure how you receive alerts and updates
                  </p>
                </div>
                <Separator />
                <div className="space-y-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Email Notifications</p>
                      <p className="text-sm text-muted-foreground">
                        Receive notifications via email
                      </p>
                    </div>
                    <Switch
                      checked={notifications.email}
                      onCheckedChange={(checked) =>
                        setNotifications((prev) => ({ ...prev, email: checked }))
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Deadline Alerts</p>
                      <p className="text-sm text-muted-foreground">
                        Get notified about approaching RFP deadlines
                      </p>
                    </div>
                    <Switch
                      checked={notifications.deadlineAlerts}
                      onCheckedChange={(checked) =>
                        setNotifications((prev) => ({ ...prev, deadlineAlerts: checked }))
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Agent Activity Updates</p>
                      <p className="text-sm text-muted-foreground">
                        Receive updates when AI agents complete tasks
                      </p>
                    </div>
                    <Switch
                      checked={notifications.agentUpdates}
                      onCheckedChange={(checked) =>
                        setNotifications((prev) => ({ ...prev, agentUpdates: checked }))
                      }
                    />
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Weekly Digest</p>
                      <p className="text-sm text-muted-foreground">
                        Summary of all RFP activity sent weekly
                      </p>
                    </div>
                    <Switch
                      checked={notifications.weeklyDigest}
                      onCheckedChange={(checked) =>
                        setNotifications((prev) => ({ ...prev, weeklyDigest: checked }))
                      }
                    />
                  </div>
                </div>
              </motion.div>
            </TabsContent>

            {/* AI Agents Tab */}
            <TabsContent value="agents">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-card rounded-xl border border-border p-6 space-y-6"
              >
                <div>
                  <h3 className="font-semibold mb-1">AI Agent Configuration</h3>
                  <p className="text-sm text-muted-foreground">
                    Configure autonomous agent behavior
                  </p>
                </div>
                <Separator />
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-6">
                    <div className="space-y-2">
                      <Label>Sales Agent - Scan Frequency</Label>
                      <Select defaultValue="hourly">
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="realtime">Real-time</SelectItem>
                          <SelectItem value="hourly">Every Hour</SelectItem>
                          <SelectItem value="daily">Daily</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label>Auto-Analyze New RFPs</Label>
                      <Select defaultValue="enabled">
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="enabled">Enabled</SelectItem>
                          <SelectItem value="disabled">Disabled</SelectItem>
                          <SelectItem value="prompt">Prompt First</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                  <div className="space-y-2">
                    <Label>Minimum Spec-Match Threshold for Auto-Pricing</Label>
                    <Select defaultValue="70">
                      <SelectTrigger className="w-48">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="90">90% or higher</SelectItem>
                        <SelectItem value="80">80% or higher</SelectItem>
                        <SelectItem value="70">70% or higher</SelectItem>
                        <SelectItem value="60">60% or higher</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="p-4 bg-info/10 border border-info/30 rounded-lg">
                    <p className="text-sm text-info">
                      <strong>Note:</strong> Agents operate autonomously but require human approval for final bid submission.
                    </p>
                  </div>
                </div>
              </motion.div>
            </TabsContent>

            {/* Security Tab */}
            <TabsContent value="security">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-card rounded-xl border border-border p-6 space-y-6"
              >
                <div>
                  <h3 className="font-semibold mb-1">Security Settings</h3>
                  <p className="text-sm text-muted-foreground">
                    Manage your account security
                  </p>
                </div>
                <Separator />
                <div className="space-y-6">
                  <div className="space-y-4">
                    <Label>Change Password</Label>
                    <div className="grid gap-4">
                      <Input type="password" placeholder="Current Password" />
                      <Input type="password" placeholder="New Password" />
                      <Input type="password" placeholder="Confirm New Password" />
                    </div>
                    <Button>Update Password</Button>
                  </div>
                  <Separator />
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Two-Factor Authentication</p>
                      <p className="text-sm text-muted-foreground">
                        Add an extra layer of security to your account
                      </p>
                    </div>
                    <Button variant="outline">Enable 2FA</Button>
                  </div>
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-medium">Active Sessions</p>
                      <p className="text-sm text-muted-foreground">
                        Manage your active login sessions
                      </p>
                    </div>
                    <Button variant="outline">View Sessions</Button>
                  </div>
                </div>
              </motion.div>
            </TabsContent>
          </Tabs>
        </motion.div>
      </div>
    </MainLayout>
  );
}
