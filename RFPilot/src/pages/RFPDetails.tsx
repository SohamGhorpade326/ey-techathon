import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { useParams } from "react-router-dom";
import Groq from "groq-sdk";
import {
  Calendar,
  Building2,
  AlertTriangle,
  ExternalLink,
  Package,
  FileText,
  DollarSign,
  Play,
  Cpu,
  Loader2,
  Sparkles,
  Download,
} from "lucide-react";

import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";

// Initialize Groq with your provided key
const groq = new Groq({ 
  apiKey: import.meta.env.VITE_GROQ_API_KEY,
  dangerouslyAllowBrowser: true 
});

export default function RFPDetails() {
  const [rfp, setRfp] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<string | null>(null);

  const { rfpId } = useParams();
  const decodedId = decodeURIComponent(rfpId || "");

  // In RFPDetails.tsx
useEffect(() => {
  if (!rfpId) return;
  setLoading(true);
  
  // Use raw rfpId from useParams directly
  const fetchUrl = `http://127.0.0.1:8000/rfps/${rfpId}`;

  fetch(fetchUrl)
    .then(async (res) => {
      const data = await res.json();
      if (data.error) throw new Error(data.error);
      setRfp(data);
      setError(null);
    })
    .catch((err) => {
      setError(err.message);
      setRfp(null);
    })
    .finally(() => setLoading(false));
}, [rfpId]);


  const handleAnalyzeRFP = async () => {
    if (!rfp) return;
    setIsAnalyzing(true);
    setSummary(null);

    try {
      const chatCompletion = await groq.chat.completions.create({
        messages: [
          {
            role: "system",
            content: "You are a senior sales analyst. Provide professional, concise RFP summaries.",
          },
          {
            role: "user",
            // Prompt updated to match new field keys from backend
            content: `Summarize this RFP in professional language. Focus on: 1) Executive Summary, 2) Key Deliverables, and 3) Deadline Urgency. Use bullet points.

            RFP Data:
            Title: ${rfp.rfp_title}
            Buyer: ${rfp.buyer}
            Deadline: ${rfp.submission_deadline}
            Value: ${rfp.estimated_project_value}
            Priority: ${rfp.priority}`,
          
          },
        ],
        model: "llama-3.3-70b-versatile",
        temperature: 0.5,
      });

      const text = chatCompletion.choices[0]?.message?.content;
      if (!text) throw new Error("Empty response from Groq");
      setSummary(text);
    } catch (err: any) {
      console.error("Groq API Error:", err);
      setSummary(`Analysis Failed: ${err.message || "Check API key or Rate Limits"}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleDownloadSummary = () => {
    const printableWindow = window.open('', '_blank');
    if (!printableWindow) return;

    printableWindow.document.write(`
      <html>
        <head>
          <title>RFP Intelligence Report - ${rfp.rfp_id}</title>
          <style>
            body { font-family: 'Segoe UI', sans-serif; padding: 40px; color: #1a1a1a; line-height: 1.6; }
            .header { border-bottom: 3px solid #10b981; padding-bottom: 10px; margin-bottom: 20px; }
            h1 { color: #10b981; margin: 0; }
            .meta { background: #f3f4f6; padding: 15px; border-radius: 8px; margin-bottom: 30px; font-size: 14px; }
            .content { white-space: pre-wrap; font-size: 16px; }
          </style>
        </head>
        <body>
          <div class="header"><h1>RFP Intelligence Report</h1></div>
          <div class="meta">
            <strong>ID:</strong> ${rfp.rfp_id}<br/>
            <strong>Title:</strong> ${rfp.rfp_title}<br/>
            <strong>Buyer:</strong> ${rfp.buyer}<br/>
            <strong>Generated:</strong> ${new Date().toLocaleDateString()}
          </div>
          <div class="content">${summary}</div>
        </body>
      </html>
    `);
    printableWindow.document.close();
    printableWindow.print();
  };

  if (loading) return (
    <MainLayout title="Loading RFP...">
      <div className="flex justify-center py-20"><Loader2 className="h-8 w-8 animate-spin text-primary" /></div>
    </MainLayout>
  );

  if (!rfp || rfp.error) return (
    <MainLayout title="RFP Not Found">
      <div className="text-center py-20">
        <p className="text-muted-foreground mb-4">{error || "RFP data not available"}</p>
        <Button onClick={() => window.location.reload()}>Retry Connection</Button>
      </div>
    </MainLayout>
  );

  return (
    <MainLayout title={rfp.rfp_title}>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <motion.div className="lg:col-span-2 card-elevated p-6">
          <div className="flex justify-between">
            <div>
              <p className="text-sm font-mono text-muted-foreground">{decodedId}</p>
              <h1 className="text-2xl font-bold mt-1">{rfp.rfp_title}</h1>
            </div>
            <Button size="icon" variant="ghost" asChild>
              <a href={rfp.tender_source} target="_blank" rel="noreferrer"><ExternalLink className="h-5 w-5" /></a>
            </Button>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-6 mt-6 pt-6 border-t">
            <Info label="Buyer" icon={<Building2 />} value={rfp.buyer} />
            <Info label="Scope Items" icon={<Package />} value={`${rfp.scope_items}`} />
            <Info label="Est. Value" icon={<DollarSign />} value={rfp.estimated_project_value} />
            <Info label="Status" icon={<FileText />} value={rfp.status} />
            <Info label="Days Remaining" icon={<Calendar />} value={`${rfp.days_remaining} days`} />
          </div>
        </motion.div>

        <motion.div className="card-elevated p-6 border-2 border-red-500 bg-red-500/5">
          <div className="flex items-center gap-2 mb-4 text-red-500">
            <AlertTriangle className="h-5 w-5" />
            <h3 className="font-semibold">Submission Deadline</h3>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-red-500">{rfp.submission_deadline}</p>
            <div className="mt-4 flex justify-center">
              <span className={`px-4 py-1 rounded-full text-sm font-semibold ${rfp.priority === "High" ? "bg-red-600 text-white" : "bg-yellow-400 text-black"}`}>
                Priority: {rfp.priority}
              </span>
            </div>
          </div>
        </motion.div>
      </div>

      <AnimatePresence>
        {summary && (
          <motion.div 
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            className="mt-6 border-2 border-green-500 bg-green-500/10 rounded-xl overflow-hidden"
          >
            <div className="flex items-center justify-between bg-green-500/20 p-4 border-b border-green-500">
              <div className="flex items-center gap-2 text-green-500 font-bold">
                <Sparkles className="h-5 w-5" />
                <h3>AI Strategic Summary (Groq)</h3>
              </div>
              {!summary.includes("Failed") && (
                <Button 
                  variant="outline" 
                  size="sm" 
                  onClick={handleDownloadSummary}
                  className="bg-green-600 hover:bg-green-700 text-white border-none h-8"
                >
                  <Download className="h-4 w-4 mr-2" />
                  Download Report
                </Button>
              )}
            </div>
            <div className="p-6 text-sm leading-relaxed text-green-50 whitespace-pre-wrap">
              {summary}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="card-elevated p-4 mt-6 flex gap-4">
        <Button
          size="lg"
          onClick={handleAnalyzeRFP}
          disabled={isAnalyzing}
          className={summary && !summary.includes("Failed") ? "bg-green-600 hover:bg-green-700 text-white" : ""}
        >
          {isAnalyzing ? (
            <><Loader2 className="h-4 w-4 animate-spin mr-2" /> Generating Insights...</>
          ) : (
            <><Play className="h-4 w-4 mr-2" /> {summary ? "Refresh Analysis" : "Analyze Full RFP (AI)"}</>
          )}
        </Button>
        <Button variant="outline"><Cpu className="h-4 w-4 mr-2" /> Technical Specs</Button>
        <Button variant="outline"><DollarSign className="h-4 w-4 mr-2" /> Pricing Intel</Button>
      </div>
      
    </MainLayout>
  );
}

function Info({ label, value, icon }: any) {
  return (
    <div>
      <p className="text-xs text-muted-foreground uppercase mb-1">{label}</p>
      <div className="flex items-center gap-2">
        <div className="text-primary h-4 w-4">{icon}</div>
        <span className="font-medium text-sm">{value}</span>
      </div>
    </div>
  );
}