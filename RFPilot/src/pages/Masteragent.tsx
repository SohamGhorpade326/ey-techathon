import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import Groq from "groq-sdk";
import { 
  Cpu, 
  DollarSign, 
  ShieldCheck, 
  ClipboardList, 
  Settings, 
  Download, 
  Sparkles, 
  Loader2,
  ArrowLeft,
  GanttChart,
  CheckCircle2,
  ListTodo,
  Boxes
} from "lucide-react";
import { useNavigate } from "react-router-dom";

import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";

const groq = new Groq({ 
  apiKey: "",
  dangerouslyAllowBrowser: true 
});

export default function Masteragent() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [aiSummary, setAiSummary] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const navigate = useNavigate();

  useEffect(() => {
    fetch("http://127.0.0.1:8000/master/output")
      .then((res) => res.json())
      .then((json) => {
        if (!json.error) setData(json);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const generateAISummary = async () => {
    if (!data) return;
    setIsAnalyzing(true);
    try {
      const completion = await groq.chat.completions.create({
        messages: [
          { role: "system", content: "You are a senior RFP strategist. Analyze technical requirements, testing protocols, and pricing constraints to provide a 3-4 paragraph executive overview. Focus on technical-financial alignment and compliance risks." },
          { role: "user", content: `Analyze this Master Agent payload: ${JSON.stringify(data)}` }
        ],
        model: "llama-3.3-70b-versatile",
      });
      setAiSummary(completion.choices[0]?.message?.content || "");
    } catch (err) {
      setAiSummary("Failed to generate strategic overview.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  if (loading) return <div className="flex h-screen items-center justify-center"><Loader2 className="animate-spin text-primary" /></div>;
  if (!data) return <div className="p-20 text-center">Master Agent data not found. Please execute Sales Agent first.</div>;

  const { technical_summary: tech, pricing_summary: price } = data;

  return (
    <MainLayout title="Master Agent Coordination">
      <div className="mb-6 flex items-center justify-between">
        <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>
          <ArrowLeft className="mr-2 h-4 w-4" /> Back to Dashboard
        </Button>
        <div className="flex gap-3">
          {/* PURPLE GRADIENT BUTTON */}
          <Button 
            size="sm" 
            onClick={generateAISummary} 
            disabled={isAnalyzing}
            className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white border-none shadow-lg shadow-purple-500/20"
          >
            {isAnalyzing ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Sparkles className="mr-2 h-4 w-4" />}
            Generate Strategic AI Overview
          </Button>
          {aiSummary && (
            <Button variant="outline" size="sm" onClick={() => window.print()}>
              <Download className="mr-2 h-4 w-4" /> Export Report
            </Button>
          )}
        </div>
      </div>

      <div className="space-y-8 pb-10">
        
        {/* 1. TECHNICAL COORDINATION LAYER */}
        <section className="card-elevated border-t-4 border-blue-600 bg-slate-900/40 overflow-hidden">
          <div className="p-6 bg-blue-600/5 flex items-center gap-3 border-b border-white/5">
            <Cpu className="text-blue-400 h-6 w-6" />
            <h2 className="text-xl font-bold tracking-tight text-white">Technical Coordination Layer</h2>
          </div>
          
          <div className="p-6">
            {/* Row 1: Overview & Context */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              {/* Project Overview */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest">Project Overview</h3>
                <div className="p-5 rounded-lg bg-white/5 border border-white/5">
                  <div className="grid grid-cols-2 gap-x-6 gap-y-4">
                    <DataBlock label="RFP ID" value={tech.rfp_context.rfp_id} />
                    <DataBlock label="Priority" value={tech.rfp_context.priority} />
                    <DataBlock label="Buyer" value={tech.rfp_context.buyer} />
                    <DataBlock label="Domain" value={tech.rfp_context.domain} />
                    <DataBlock label="Deadline" value={tech.rfp_context.deadline} />
                    <DataBlock label="Estimated Value" value={tech.rfp_context.estimated_value} highlight />
                  </div>
                  <Separator className="bg-white/10 my-4" />
                  <div className="grid grid-cols-1 gap-3">
                    <DataBlock label="Title" value={tech.rfp_context.title} />
                    <DataBlock label="Source" value={tech.rfp_context.source} />
                  </div>
                </div>
              </div>

              {/* Scope Context */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest flex items-center gap-2">
                  <Boxes className="h-3 w-3" /> Scope Context
                </h3>
                <div className="p-5 rounded-lg bg-white/5 border border-white/5">
                  <div className="space-y-4">
                    <DataBlock label="Material Type" value={tech.scope_context.material_type} highlight />
                    <DataBlock label="Scope Size" value={tech.scope_context.scope_size} />
                    <Separator className="bg-white/10" />
                    <div>
                      <p className="text-[10px] text-muted-foreground uppercase mb-3 font-medium">Expected Categories</p>
                      <div className="flex flex-wrap gap-2">
                        {tech.scope_context.expected_item_categories.map((cat: string) => (
                          <Badge key={cat} variant="secondary" className="text-[10px] bg-blue-500/10 text-blue-300 border-none py-1 px-3">{cat}</Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Row 2: Instructions & Matching */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Technical Instructions */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest flex items-center gap-2">
                  <ListTodo className="h-3 w-3" /> Agent Instructions
                </h3>
                <div className="p-5 rounded-lg bg-white/5 border border-white/5">
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-3">
                      <BoolCheck label="Extract Items" checked={tech.technical_instructions.extract_items} />
                      <BoolCheck label="Extract Specs" checked={tech.technical_instructions.extract_required_specs} />
                    </div>
                    <Separator className="bg-white/10" />
                    <div>
                      <p className="text-[10px] text-muted-foreground uppercase mb-3 font-medium">Specification Priority</p>
                      <div className="flex flex-wrap gap-2">
                        {tech.technical_instructions.spec_priority.map((spec: string) => (
                          <Badge key={spec} variant="outline" className="text-[9px] border-blue-500/30 text-blue-400 font-mono uppercase py-1 px-2">{spec}</Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* SKU Matching Rules */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-blue-400 uppercase tracking-widest flex items-center gap-2">
                  <GanttChart className="h-3 w-3" /> SKU Matching Rules
                </h3>
                <div className="p-5 rounded-lg bg-white/5 border border-white/5">
                  <div className="grid grid-cols-2 gap-4">
                    <MetricCard label="Min Threshold" value={`${tech.sku_matching_rules.min_match_threshold}%`} />
                    <MetricCard label="Match Limit" value={`Top ${tech.sku_matching_rules.top_n}`} />
                    <MetricCard label="Warning Band" value={`≥${tech.sku_matching_rules.warning_threshold}%`} warning />
                    <MetricCard label="Green Band" value={`≥${tech.sku_matching_rules.green_threshold}%`} success />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 2. FINANCIAL & TESTING LAYER */}
        <section className="card-elevated border-t-4 border-emerald-600 bg-slate-900/40 overflow-hidden">
          <div className="p-6 bg-emerald-600/5 flex items-center gap-3 border-b border-white/5">
            <DollarSign className="text-emerald-400 h-6 w-6" />
            <h2 className="text-xl font-bold tracking-tight text-white">Financial & Testing Layer</h2>
          </div>
          
          <div className="p-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Material Scope */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-widest">Material Scope</h3>
                <div className="p-5 rounded-lg bg-white/5 border border-white/5">
                  <div className="space-y-4">
                    <DataBlock label="Material Type" value={price.material_scope.material_type} />
                    <Separator className="bg-white/10" />
                    <div>
                      <p className="text-[10px] text-muted-foreground uppercase mb-3 font-medium">Expected Categories</p>
                      <div className="flex flex-wrap gap-2">
                        {price.material_scope.expected_categories.map((cat: string) => (
                          <Badge key={cat} variant="secondary" className="text-[10px] bg-emerald-500/10 text-emerald-300 border-none py-1 px-3">{cat}</Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Compliance Protocols */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-widest">Compliance Protocols</h3>
                <div className="p-5 rounded-lg bg-white/5 border border-white/5">
                  <div className="space-y-4">
                    <div>
                      <p className="text-[10px] text-muted-foreground uppercase mb-3 font-medium">Testing Requirements</p>
                      <div className="grid grid-cols-1 gap-2">
                        <BoolCheck label="Routine Tests" checked={price.testing_requirements.include_routine_tests} />
                        <BoolCheck label="Type Tests" checked={price.testing_requirements.include_type_tests} />
                        <BoolCheck label="Acceptance Tests" checked={price.testing_requirements.include_acceptance_tests} />
                      </div>
                    </div>
                    <Separator className="bg-white/10" />
                    <div>
                      <p className="text-[10px] text-muted-foreground uppercase mb-3 font-medium">Standards</p>
                      <div className="flex flex-wrap gap-2">
                        {price.testing_requirements.standards.map((std: string) => (
                          <Badge key={std} variant="outline" className="text-[10px] border-emerald-500/30 text-emerald-400 font-mono py-1 px-3">{std}</Badge>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Pricing Algorithms */}
              <div className="space-y-4">
                <h3 className="text-xs font-bold text-emerald-400 uppercase tracking-widest">Pricing Rules</h3>
                <div className="p-5 rounded-lg bg-white/5 border border-white/5">
                  <div className="space-y-4">
                    <div className="grid grid-cols-2 gap-x-4 gap-y-3">
                      <DataBlock label="Currency" value={price.pricing_rules.currency} />
                      <DataBlock label="Risk Buffer" value={`${price.pricing_rules.risk_threshold_percent}%`} />
                      <div className="col-span-2">
                        <DataBlock label="Material Source" value={price.pricing_rules.material_price_source} />
                      </div>
                      <div className="col-span-2">
                        <DataBlock label="Testing Source" value={price.pricing_rules.testing_price_source} />
                      </div>
                    </div>
                    <Separator className="bg-white/10" />
                    <div className="flex flex-col items-center p-4 bg-emerald-500/10 rounded-lg">
                      <p className="text-[10px] text-emerald-400 uppercase mb-2 font-bold tracking-wider">Agent Status</p>
                      <p className="text-sm font-bold text-white uppercase tracking-wide">{price.status}</p>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* PDF PATH INFO */}
        

        {/* AI STRATEGY OVERVIEW */}
        <AnimatePresence>
          {aiSummary && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="card-elevated p-8 bg-gradient-to-br from-purple-500/10 to-indigo-500/10 border border-purple-500/30"
            >
              <div className="flex items-center gap-2 mb-4 text-purple-400 font-bold">
                <Sparkles className="h-5 w-5" />
                <h3>Strategic Executive Overview</h3>
              </div>
              <div className="text-sm leading-relaxed text-slate-200 prose prose-invert max-w-none whitespace-pre-wrap">
                {aiSummary}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </MainLayout>
  );
}

// UI HELPER COMPONENTS
function DataBlock({ label, value, highlight = false }: any) {
  return (
    <div>
      <p className="text-[10px] text-muted-foreground uppercase font-medium tracking-wider mb-1">{label}</p>
      <p className={`text-sm font-semibold leading-tight ${highlight ? "text-primary" : "text-white"}`}>{value || "N/A"}</p>
    </div>
  );
}

function Row({ label, value }: any) {
  return (
    <tr className="hover:bg-white/5 transition-colors">
      <td className="py-2 px-1 text-muted-foreground font-medium">{label}</td>
      <td className="py-2 px-1 text-right text-white font-mono">{value}</td>
    </tr>
  );
}

function MetricCard({ label, value, success, warning }: any) {
  return (
    <div className={`p-3 rounded-lg border bg-slate-900/40 ${success ? "border-emerald-500/50" : warning ? "border-yellow-500/50" : "border-white/10"}`}>
      <p className="text-[9px] uppercase text-muted-foreground mb-1 font-bold tracking-tighter">{label}</p>
      <p className={`text-base font-bold ${success ? "text-emerald-400" : warning ? "text-yellow-400" : "text-white"}`}>{value}</p>
    </div>
  );
}

function BoolCheck({ label, checked }: any) {
  return (
    <div className="flex items-center gap-2 px-2 py-1.5 rounded bg-white/5 border border-white/5">
      <CheckCircle2 className={`h-3 w-3 ${checked ? "text-emerald-400" : "text-slate-600"}`} />
      <span className={`text-[10px] font-medium ${checked ? "text-slate-200" : "text-slate-500"}`}>{label}</span>
    </div>
  );
}