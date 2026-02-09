import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableHeader,
  TableRow,
  TableHead,
  TableBody,
  TableCell,
} from "@/components/ui/table";
import { Eye, Trash2 } from "lucide-react";

const API = "http://127.0.0.1:8000"; // must match backend port

export default function Discovery() {
  const [rfps, setRfps] = useState<any[]>([]);
  const [search, setSearch] = useState("");
  const [isRunning, setIsRunning] = useState(false);
  const navigate = useNavigate();

  // -----------------------------
  // Fetch RFPS
  // -----------------------------
  const fetchRFPs = async () => {
    try {
      const res = await fetch(`${API}/rfps`);
      const data = await res.json();
      
      // Ensure data is an array
      if (Array.isArray(data)) {
        setRfps(data);
        return data.length;
      } else {
        setRfps([]);
        return 0;
      }
    } catch (error) {
      console.error("Error fetching RFPs:", error);
      setRfps([]);
      return 0;
    }
  };

  // -----------------------------
  // Run Sales Agent + Poll
  // -----------------------------
  // Inside Discovery.tsx -> runSalesAgent function
const runSalesAgent = async () => {
  setIsRunning(true);

  // 1. Start the agent and get the task_id
  const response = await fetch(`${API}/run`, { method: "POST" });
  const { task_id } = await response.json();

  // 2. Start polling
  const poll = setInterval(async () => {
    // Refresh the table with whatever is currently in the store
    await fetchRFPs();

    // 3. Check the actual background task status
    try {
      const statusRes = await fetch(`${API}/run/status/${task_id}`);
      const { status } = await statusRes.json();

      // Only stop polling when the backend is finished or failed
      if (status === "completed" || status === "failed") {
        clearInterval(poll);
        setIsRunning(false);
        // Final fetch to ensure all data is caught
        await fetchRFPs(); 
      }
    } catch (err) {
      console.error("Status check failed", err);
    }
  }, 12000); // Poll every 2 seconds
};

  // -----------------------------
  // Clear RFPS
  // -----------------------------
  const deleteAllRFPs = async () => {
    if (!confirm("Delete all discovered RFPs?")) return;
    await fetch(`${API}/rfps`, { method: "DELETE" });
    setRfps([]);
  };

  useEffect(() => {
    fetchRFPs();
  }, []);

  // -----------------------------
  // Filter
  // -----------------------------
  const filtered = rfps.filter(
    (r) =>
      r.title?.toLowerCase().includes(search.toLowerCase()) ||
      r.buyer?.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <MainLayout title="RFP Discovery">
      {/* HEADER */}
      <div className="flex justify-between items-center mb-6 gap-4 p-4 bg-slate-900/40 rounded-lg border border-white/10">
        <Input
          placeholder="🔍 Search by title or buyer..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-1/3 bg-slate-800/50 border-white/10 focus:border-blue-500/50"
        />

        <div className="flex gap-3">
          <Button 
            variant="destructive" 
            onClick={deleteAllRFPs}
            className="bg-red-600 hover:bg-red-700 border-none shadow-lg shadow-red-500/20"
          >
            <Trash2 className="h-4 w-4 mr-2" />
            Clear All
          </Button>

          <Button 
            onClick={runSalesAgent} 
            disabled={isRunning}
            className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white border-none shadow-lg shadow-blue-500/20"
          >
            {isRunning ? (
              <>
                <div className="mr-2 h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"></div>
                Discovering...
              </>
            ) : (
              "Run Sales Agent"
            )}
          </Button>
        </div>
      </div>

      {/* STATS SUMMARY */}
      {rfps.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          <div className="p-4 rounded-lg bg-gradient-to-br from-blue-600/10 to-blue-500/5 border border-blue-500/20">
            <p className="text-xs text-blue-400 uppercase tracking-wider mb-1">Total RFPs</p>
            <p className="text-2xl font-bold text-white">{rfps.length}</p>
          </div>
          <div className="p-4 rounded-lg bg-gradient-to-br from-red-600/10 to-red-500/5 border border-red-500/20">
            <p className="text-xs text-red-400 uppercase tracking-wider mb-1">Critical</p>
            <p className="text-2xl font-bold text-white">
              {rfps.filter(r => r.priority === "Critical").length}
            </p>
          </div>
          <div className="p-4 rounded-lg bg-gradient-to-br from-orange-600/10 to-orange-500/5 border border-orange-500/20">
            <p className="text-xs text-orange-400 uppercase tracking-wider mb-1">High Priority</p>
            <p className="text-2xl font-bold text-white">
              {rfps.filter(r => r.priority === "High").length}
            </p>
          </div>
          <div className="p-4 rounded-lg bg-gradient-to-br from-emerald-600/10 to-emerald-500/5 border border-emerald-500/20">
            <p className="text-xs text-emerald-400 uppercase tracking-wider mb-1">Extracted</p>
            <p className="text-2xl font-bold text-white">
              {rfps.filter(r => r.status === "Extracted").length}
            </p>
          </div>
        </div>
      )}

      {/* TABLE */}
      <div className="rounded-xl border border-white/10 bg-slate-900/40 overflow-hidden shadow-2xl">
        <Table>
          <TableHeader>
            <TableRow className="bg-gradient-to-r from-blue-600/20 via-indigo-600/20 to-purple-600/20 border-b border-white/10 hover:bg-gradient-to-r hover:from-blue-600/30 hover:via-indigo-600/30 hover:to-purple-600/30">
              <TableHead className="text-blue-300 font-bold text-xs uppercase tracking-wider">ID</TableHead>
              <TableHead className="text-blue-300 font-bold text-xs uppercase tracking-wider">Title</TableHead>
              <TableHead className="text-blue-300 font-bold text-xs uppercase tracking-wider">Deadline</TableHead>
              <TableHead className="text-blue-300 font-bold text-xs uppercase tracking-wider">Est. Value</TableHead>
              <TableHead className="text-blue-300 font-bold text-xs uppercase tracking-wider">Domain</TableHead>
              <TableHead className="text-blue-300 font-bold text-xs uppercase tracking-wider">Priority</TableHead>
              <TableHead className="text-blue-300 font-bold text-xs uppercase tracking-wider">Status</TableHead>
              <TableHead className="text-center text-blue-300 font-bold text-xs uppercase tracking-wider">Action</TableHead>
            </TableRow>
          </TableHeader>

          <TableBody>
            {filtered.map((rfp, index) => (
              <TableRow 
                key={rfp.rfp_id}
                className={`border-b border-white/5 hover:bg-white/5 transition-colors ${
                  index % 2 === 0 ? 'bg-slate-800/20' : 'bg-slate-800/10'
                }`}
              >
                <TableCell className="font-mono text-xs text-slate-300 py-4">
                  <div className="px-2 py-1 bg-slate-700/30 rounded border border-white/10 inline-block">
                    {rfp.rfp_id}
                  </div>
                </TableCell>

                <TableCell className="font-medium text-sm text-white max-w-md">
                  <div className="line-clamp-2">{rfp.rfp_title}</div>
                </TableCell>
               
                <TableCell className="text-sm text-slate-300">
                  <div className="flex items-center gap-2">
                    <span className="text-xs px-2 py-1 bg-orange-500/10 text-orange-300 rounded border border-orange-500/20">
                      {rfp.submission_deadline}
                    </span>
                  </div>
                </TableCell>

                <TableCell className="text-sm font-semibold text-emerald-400">
                  {rfp.estimated_project_value}
                </TableCell>
               
                <TableCell>
                  <Badge 
                    variant="outline" 
                    className="text-xs border-cyan-500/30 text-cyan-300 bg-cyan-500/10 font-medium"
                  >
                    {rfp.domain}
                  </Badge>
                </TableCell>

                {/* PRIORITY */}
                <TableCell>
                  <Badge
                    variant="outline"
                    className={`text-xs font-bold ${
                      rfp.priority === "Critical"
                        ? "border-red-500 text-red-400 bg-red-500/10"
                        : rfp.priority === "High"
                        ? "border-orange-500 text-orange-400 bg-orange-500/10"
                        : "border-yellow-500 text-yellow-400 bg-yellow-500/10"
                    }`}
                  >
                    {rfp.priority}
                  </Badge>
                </TableCell>

                {/* STATUS */}
                <TableCell>
                  <Badge 
                    variant="secondary" 
                    className="text-xs bg-blue-500/10 text-blue-300 border border-blue-500/20 font-medium"
                  >
                    {rfp.status}
                  </Badge>
                </TableCell>

                <TableCell className="text-center">
                  <Button
                    size="icon"
                    variant="ghost"
                    className="hover:bg-blue-500/20 hover:text-blue-400"
                    onClick={() =>
  navigate(`/rfp/${encodeURIComponent(rfp.rfp_id)}`)


}

                  >
                    <Eye className="h-4 w-4" />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>

        {!isRunning && rfps.length === 0 && (
          <div className="text-center py-16 bg-slate-800/20">
            <p className="text-muted-foreground text-lg mb-2">No RFPs discovered yet</p>
            <p className="text-sm text-slate-500">Click "Run Sales Agent" to start discovering RFPs</p>
          </div>
        )}
        
        {isRunning && rfps.length === 0 && (
          <div className="text-center py-16 bg-slate-800/20">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mb-4"></div>
            <p className="text-blue-400 text-lg font-medium">Discovering RFPs...</p>
            <p className="text-sm text-slate-500 mt-2">Please wait while we scan for new opportunities</p>
          </div>
        )}
      </div>
    </MainLayout>
  );
}
