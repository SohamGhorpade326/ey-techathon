import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import {
  TrendingUp,
  AlertTriangle,
  Package,
  DollarSign,
  Loader2,
  Sparkles,
  CheckCircle2,
  XCircle,
  AlertCircle,
  ArrowLeft,
} from "lucide-react";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

const API_BASE = "http://localhost:5000/api/sku-gap";

interface Summary {
  total_items: number;
  avg_match_percent: number;
  full_count: number;
  partial_count: number;
  incomplete_count: number;
  total_missing_specs: number;
}

interface DomainAnalysis {
  domain: string;
  total_gaps: number;
  avg_match: number;
}

interface SpecFrequency {
  spec_name: string;
  count: number;
}

interface MatchDistribution {
  high: number;
  medium: number;
  low: number;
}

interface Recommendation {
  spec_name: string;
  frequency: number;
  suggestion: string;
  impact_level: string;
}

interface Timeline {
  date: string;
  avg_match: number;
}

interface RiskValue {
  risk_items: number;
  estimated_risk_score: number;
}

interface Insights {
  insights_text: string;
  data_summary: {
    total_items: number;
    avg_match: number;
    total_gaps: number;
  };
}

export default function SkuGapAnalytics() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<Summary | null>(null);
  const [domainAnalysis, setDomainAnalysis] = useState<DomainAnalysis[]>([]);
  const [specFrequency, setSpecFrequency] = useState<SpecFrequency[]>([]);
  const [matchDistribution, setMatchDistribution] = useState<MatchDistribution | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [timeline, setTimeline] = useState<Timeline[]>([]);
  const [riskValue, setRiskValue] = useState<RiskValue | null>(null);
  const [insights, setInsights] = useState<Insights | null>(null);

  useEffect(() => {
    fetchAllData();
  }, []);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      const token = localStorage.getItem("token");
      const headers = {
        Authorization: `Bearer ${token}`,
      };
      const [
        summaryRes,
        domainRes,
        specRes,
        matchRes,
        recoRes,
        timeRes,
        riskRes,
        insightsRes,
      ] = await Promise.all([
        fetch(`${API_BASE}/summary`, { headers }),
        fetch(`${API_BASE}/domain-analysis`, { headers }),
        fetch(`${API_BASE}/spec-frequency`, { headers }),
        fetch(`${API_BASE}/match-distribution`, { headers }),
        fetch(`${API_BASE}/recommendations`, { headers }),
        fetch(`${API_BASE}/timeline`, { headers }),
        fetch(`${API_BASE}/risk-value`, { headers }),
        fetch(`${API_BASE}/insights`, { headers }),
      ]);

      const summaryData = await summaryRes.json();
      const domainData = await domainRes.json();
      const specData = await specRes.json();
      const matchData = await matchRes.json();
      const recoData = await recoRes.json();
      const timeData = await timeRes.json();
      const riskData = await riskRes.json();
      const insightsData = await insightsRes.json();

      setSummary(summaryData);
      setDomainAnalysis(domainData);
      setSpecFrequency(specData.slice(0, 10)); // Top 10
      setMatchDistribution(matchData);
      setRecommendations(recoData);
      setTimeline(timeData);
      setRiskValue(riskData);
      setInsights(insightsData);
    } catch (error) {
      console.error("Error fetching SKU gap data:", error);
    } finally {
      setLoading(false);
    }
  };

  const formatCurrency = (value: number) => {
    return `₹${value.toLocaleString("en-IN")}`;
  };

  const formatPercent = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="text-center"
        >
          <Loader2 className="h-12 w-12 animate-spin text-primary mx-auto mb-4" />
          <p className="text-muted-foreground">Loading SKU Gap Intelligence...</p>
        </motion.div>
      </div>
    );
  }

  const pieColors = ["#10b981", "#f59e0b", "#ef4444"];
  const pieData = matchDistribution
    ? [
        { name: "High (≥80%)", value: matchDistribution.high, color: pieColors[0] },
        { name: "Medium (60-79%)", value: matchDistribution.medium, color: pieColors[1] },
        { name: "Low (<60%)", value: matchDistribution.low, color: pieColors[2] },
      ]
    : [];

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: { staggerChildren: 0.1 },
    },
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 },
  };

  return (
    <div className="min-h-screen bg-background p-6 space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div className="flex items-center gap-4">
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => navigate("/dashboard")}
            className="p-2 rounded-lg bg-card border border-border hover:bg-accent transition-colors"
          >
            <ArrowLeft className="h-5 w-5 text-foreground" />
          </motion.button>
          <div>
            <h1 className="text-4xl font-bold text-foreground flex items-center gap-3">
              <Package className="h-10 w-10 text-primary" />
              SKU Gap Intelligence
            </h1>
            <p className="text-muted-foreground mt-1">
              Comprehensive analysis of product specification coverage
            </p>
          </div>
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={fetchAllData}
          className="px-4 py-2 bg-primary text-primary-foreground rounded-lg font-medium hover:bg-primary/90 transition-colors"
        >
          Refresh Data
        </motion.button>
      </motion.div>

      {/* Row 1: KPI Cards */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6"
      >
        <motion.div variants={item}>
          <KPICard
            title="Total SKU Items"
            value={summary?.total_items.toString() || "0"}
            icon={Package}
            gradient="from-blue-500 to-blue-700"
          />
        </motion.div>
        <motion.div variants={item}>
          <KPICard
            title="Avg Match Percentage"
            value={formatPercent(summary?.avg_match_percent || 0)}
            icon={TrendingUp}
            gradient="from-green-500 to-green-700"
          />
        </motion.div>
        <motion.div variants={item}>
          <KPICard
            title="Total Missing Specs"
            value={summary?.total_missing_specs.toString() || "0"}
            icon={AlertTriangle}
            gradient="from-orange-500 to-orange-700"
          />
        </motion.div>
        <motion.div variants={item}>
          <KPICard
            title="Estimated Risk Value"
            value={formatCurrency(riskValue?.estimated_risk_score || 0)}
            icon={DollarSign}
            gradient="from-red-500 to-red-700"
          />
        </motion.div>
      </motion.div>

      {/* Row 2: Domain Analysis & Match Distribution */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <motion.div variants={item}>
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="text-foreground">Domain Gap Analysis</CardTitle>
              <CardDescription>Missing specifications by product domain</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={domainAnalysis}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="domain" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar dataKey="total_gaps" fill="#3b82f6" radius={[8, 8, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="text-foreground">Match Distribution</CardTitle>
              <CardDescription>SKU coverage quality breakdown</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, value }) => `${name}: ${value}`}
                    outerRadius={100}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>

      {/* Row 3: Spec Frequency & Timeline */}
      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 lg:grid-cols-2 gap-6"
      >
        <motion.div variants={item}>
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="text-foreground">Most Missing Specifications</CardTitle>
              <CardDescription>Top 10 frequently missing specs</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={specFrequency} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis type="number" stroke="hsl(var(--muted-foreground))" />
                  <YAxis dataKey="spec_name" type="category" width={150} stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Bar dataKey="count" fill="#8b5cf6" radius={[0, 8, 8, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>

        <motion.div variants={item}>
          <Card className="bg-card border-border">
            <CardHeader>
              <CardTitle className="text-foreground">Coverage Timeline</CardTitle>
              <CardDescription>Average match percentage over time</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <LineChart data={timeline}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="date" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--card))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                  <Line type="monotone" dataKey="avg_match" stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </motion.div>
      </motion.div>

      {/* Row 4: AI Insights */}
      {insights && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="bg-gradient-to-br from-purple-900/20 to-blue-900/20 border-purple-500/30">
            <CardHeader>
              <CardTitle className="text-foreground flex items-center gap-2">
                <Sparkles className="h-5 w-5 text-purple-400" />
                AI-Powered Insights
              </CardTitle>
              <CardDescription>Strategic recommendations from advanced analysis</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="prose prose-invert max-w-none">
                <pre className="whitespace-pre-wrap text-sm text-foreground/90 font-sans leading-relaxed">
                  {insights.insights_text.replace(/\*\*/g, '')}
                </pre>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      )}

      {/* Row 5: Recommendations Table */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
      >
        <Card className="bg-card border-border">
          <CardHeader>
            <CardTitle className="text-foreground">Top Recommendations</CardTitle>
            <CardDescription>Priority actions based on gap analysis</CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Specification</TableHead>
                  <TableHead>Frequency</TableHead>
                  <TableHead>Impact Level</TableHead>
                  <TableHead>Suggestion</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {recommendations.map((rec, index) => (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="border-b border-border"
                  >
                    <TableCell className="font-medium">{rec.spec_name}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{rec.frequency}</Badge>
                    </TableCell>
                    <TableCell>
                      <ImpactBadge level={rec.impact_level} />
                    </TableCell>
                    <TableCell className="text-muted-foreground">{rec.suggestion}</TableCell>
                  </motion.tr>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </motion.div>

      {/* Footer Stats */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.7 }}
        className="grid grid-cols-1 md:grid-cols-3 gap-4"
      >
        <StatCard
          label="Full Coverage"
          value={summary?.full_count || 0}
          icon={CheckCircle2}
          color="text-green-500"
        />
        <StatCard
          label="Partial Coverage"
          value={summary?.partial_count || 0}
          icon={AlertCircle}
          color="text-yellow-500"
        />
        <StatCard
          label="Incomplete Coverage"
          value={summary?.incomplete_count || 0}
          icon={XCircle}
          color="text-red-500"
        />
      </motion.div>
    </div>
  );
}

function KPICard({
  title,
  value,
  icon: Icon,
  gradient,
}: {
  title: string;
  value: string;
  icon: any;
  gradient: string;
}) {
  return (
    <Card className="bg-card border-border overflow-hidden">
      <div className={`h-2 bg-gradient-to-r ${gradient}`} />
      <CardContent className="p-6">
        <div className="flex items-center justify-between mb-2">
          <p className="text-sm text-muted-foreground">{title}</p>
          <Icon className="h-5 w-5 text-muted-foreground" />
        </div>
        <p className="text-3xl font-bold text-foreground">{value}</p>
      </CardContent>
    </Card>
  );
}

function ImpactBadge({ level }: { level: string }) {
  const variants: Record<string, { className: string; icon: any }> = {
    High: { className: "bg-red-500/20 text-red-400 border-red-500/50", icon: AlertTriangle },
    Medium: { className: "bg-yellow-500/20 text-yellow-400 border-yellow-500/50", icon: AlertCircle },
    Low: { className: "bg-blue-500/20 text-blue-400 border-blue-500/50", icon: CheckCircle2 },
  };

  const variant = variants[level] || variants.Low;
  const Icon = variant.icon;

  return (
    <Badge className={`${variant.className} border`}>
      <Icon className="h-3 w-3 mr-1" />
      {level}
    </Badge>
  );
}

function StatCard({
  label,
  value,
  icon: Icon,
  color,
}: {
  label: string;
  value: number;
  icon: any;
  color: string;
}) {
  return (
    <Card className="bg-card border-border">
      <CardContent className="p-4 flex items-center gap-4">
        <div className={`p-3 rounded-lg bg-muted ${color}`}>
          <Icon className="h-6 w-6" />
        </div>
        <div>
          <p className="text-2xl font-bold text-foreground">{value}</p>
          <p className="text-sm text-muted-foreground">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
}
