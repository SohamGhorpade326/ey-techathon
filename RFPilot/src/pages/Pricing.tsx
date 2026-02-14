import { useState, useEffect } from "react";
import { motion } from "framer-motion";
import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  AlertTriangle,
  CheckCircle2,
  DollarSign,
  Calculator,
  FileText,
  Sparkles,
  Loader2,
  Info,
  Target,
  AlertCircle,
  XCircle,
  SkipForward,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API = "http://127.0.0.1:8000";

export default function Pricing() {
  const [items, setItems] = useState<any[]>([]);
  const [pricingData, setPricingData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [showRationale, setShowRationale] = useState(false);
  const [selectedItemRationale, setSelectedItemRationale] = useState<string>("");
  const [showViabilityDialog, setShowViabilityDialog] = useState(false);
  const [isSkipping, setIsSkipping] = useState(false);

  useEffect(() => {
    fetch(`${API}/pricing/output`)
      .then((res) => res.json())
      .then((data) => {
        setPricingData(data);
        if (data.items) {
          setItems(data.items);
        }
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching pricing data:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <MainLayout title="Pricing Workspace">
        <div className="flex h-[60vh] items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </MainLayout>
    );
  }

  if (!pricingData || !items || items.length === 0) {
    return (
      <MainLayout title="Pricing Workspace">
        <div className="text-center py-20">
          <p className="text-muted-foreground">No pricing data available. Please run the Sales Agent first.</p>
        </div>
      </MainLayout>
    );
  }

  const updateQuantity = (index: number, quantity: number) => {
    setItems((prev) =>
      prev.map((item, idx) => {
        if (idx === index) {
          const newMaterialTotal = quantity * item.unit_price;
          const newLineTotal = newMaterialTotal + item.testing_cost;
          return { 
            ...item, 
            quantity, 
            material_total: newMaterialTotal,
            line_total: newLineTotal
          };
        }
        return item;
      })
    );
    
    // Update pricing data totals
    setPricingData((prev: any) => {
      if (!prev) return prev;
      const newMaterialTotal = items.reduce((sum, item, idx) => {
        if (idx === index) {
          return sum + (quantity * item.unit_price);
        }
        return sum + (item.material_total || 0);
      }, 0);
      const newGrandTotal = newMaterialTotal + (prev.testing_total || 0);
      return {
        ...prev,
        material_total: newMaterialTotal,
        grand_total: newGrandTotal
      };
    });
  };

  const materialTotal = items.reduce(
    (sum, item) => sum + (item.material_total || 0),
    0
  );
  const testingTotal = items.reduce(
    (sum, item) => sum + (item.testing_cost || 0), 
    0
  );
  const grandTotal = materialTotal + testingTotal;

  const highRiskItems = items.filter((item) => item.risk === "High");
  const mediumRiskItems = items.filter((item) => item.risk === "Medium");

  const showItemRationale = (rationale: string) => {
    setSelectedItemRationale(rationale);
    setShowRationale(true);
  };

  // Calculate Overall Spec Match % (average of all items)
  const overallSpecMatch = items.length > 0
    ? items.reduce((sum, item) => sum + (item.spec_match_percent || 0), 0) / items.length
    : 0;

  const handleSkipRFP = async () => {
    setIsSkipping(true);
    try {
      const response = await fetch(`${API}/pipeline/skip`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });

      const result = await response.json();

      if (result.status === "success") {
        alert(`RFP skipped successfully. Processing next best RFP...`);
        
        // Poll for the new task to complete
        const taskId = result.task_id;
        const pollInterval = setInterval(async () => {
          const statusRes = await fetch(`${API}/run/status/${taskId}`);
          const statusData = await statusRes.json();
          
          if (statusData.status === "completed") {
            clearInterval(pollInterval);
            // Check if there are any RFPs left
            const masterRes = await fetch(`${API}/master/output`);
            const masterData = await masterRes.json();
            
            if (masterData.error && masterData.error.includes("No RFPs available")) {
              setIsSkipping(false);
              alert("All available RFPs have been skipped. Please run the Sales Agent again to discover new opportunities.");
              window.location.href = "/dashboard";
            } else {
              // Reload the page to show the new RFP
              window.location.reload();
            }
          } else if (statusData.status === "failed") {
            clearInterval(pollInterval);
            // Check if it failed due to no RFPs
            const masterRes = await fetch(`${API}/master/output`);
            const masterData = await masterRes.json();
            
            if (masterData.error && masterData.error.includes("No RFPs available")) {
              setIsSkipping(false);
              alert("All available RFPs have been skipped. Please run the Sales Agent again to discover new opportunities.");
              window.location.href = "/dashboard";
            } else {
              setIsSkipping(false);
              alert("Failed to process next RFP. Please check the pipeline status.");
            }
          }
        }, 2000);
        
        // Timeout after 2 minutes
        setTimeout(() => {
          clearInterval(pollInterval);
          if (isSkipping) {
            setIsSkipping(false);
            alert("Pipeline is taking longer than expected. Please check the dashboard.");
          }
        }, 120000);
      } else {
        setIsSkipping(false);
        alert(`Failed to skip RFP: ${result.message}`);
      }
    } catch (error) {
      setIsSkipping(false);
      console.error("Error skipping RFP:", error);
      alert("Failed to skip RFP. Please try again.");
    }
  };

  return (
    <MainLayout
      title="Pricing Workspace"
      breadcrumbs={[
        { name: "Dashboard", href: "/dashboard" },
        { name: "Pricing & Summary" },
      ]}
    >
      <div className="space-y-6">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="flex items-center justify-between"
        >
          <div>
            <p className="text-muted-foreground">
              RFP Pricing Summary - {pricingData.currency || "INR"}
            </p>
            <div className="flex items-center gap-4 mt-2">
              <Badge variant="outline" className="bg-agent-pricing/10 text-agent-pricing border-agent-pricing/30">
                <Sparkles className="h-3 w-3 mr-1" />
                Pricing Agent Complete
              </Badge>
              <span className="text-sm text-muted-foreground">
                {items.length} items • Auto-generated from confirmed SKUs
              </span>
            </div>
          </div>
        </motion.div>

        {/* Risk Warning */}
        {(highRiskItems.length > 0 || mediumRiskItems.length > 0) && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-destructive/10 border border-destructive/30 rounded-lg p-4 flex items-start gap-3"
          >
            <AlertTriangle className="h-5 w-5 text-destructive shrink-0 mt-0.5" />
            <div>
              <p className="font-medium text-destructive">Risk Flag Detected</p>
              <p className="text-sm text-destructive/80 mt-1">
                {highRiskItems.length > 0 && `${highRiskItems.length} high-risk item(s)`}
                {highRiskItems.length > 0 && mediumRiskItems.length > 0 && " and "}
                {mediumRiskItems.length > 0 && `${mediumRiskItems.length} medium-risk item(s)`} identified. 
                Consider reviewing technical specifications or adjusting pricing accordingly.
              </p>
            </div>
          </motion.div>
        )}

        {/* Pricing Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-card rounded-xl border border-border overflow-hidden"
        >
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead>SKU Code</TableHead>
                <TableHead>Item Description</TableHead>
                <TableHead className="text-right">Quantity</TableHead>
                <TableHead className="text-right">Unit Price (₹)</TableHead>
                <TableHead className="text-right">Material Total (₹)</TableHead>
                <TableHead className="text-right">Testing Cost (₹)</TableHead>
                <TableHead className="text-right">Line Total (₹)</TableHead>
                <TableHead className="text-center">Risk</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((item, index) => {
                const isHighRisk = item.risk === "High";
                const isMediumRisk = item.risk === "Medium";

                return (
                  <motion.tr
                    key={index}
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className={cn(
                      "border-b border-border",
                      isHighRisk && "bg-destructive/5"
                    )}
                  >
                    <TableCell className="font-mono text-sm">{item.sku_code}</TableCell>
                    <TableCell>{item.item_name}</TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Input
                          type="number"
                          value={item.quantity}
                          onChange={(e) =>
                            updateQuantity(index, parseInt(e.target.value) || 0)
                          }
                          className="w-24 text-right"
                        />
                        <span className="text-xs text-muted-foreground">units</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="flex items-center justify-end gap-1">
                              <span>₹{item.unit_price?.toLocaleString()}</span>
                              <Button
                                variant="ghost"
                                size="sm"
                                className="h-5 w-5 p-0"
                                onClick={() => item.pricing_rationale && showItemRationale(item.pricing_rationale)}
                              >
                                <Info className="h-3 w-3 text-muted-foreground hover:text-primary" />
                              </Button>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent side="left" className="max-w-xs">
                            <p className="text-xs">Click info icon to view pricing justification</p>
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      <div className="flex items-center justify-end gap-1">
                        <span>₹{item.material_total?.toLocaleString()}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right font-mono">
                      <div className="flex items-center justify-end gap-1">
                        <span>₹{item.testing_cost?.toLocaleString()}</span>
                      </div>
                    </TableCell>
                    <TableCell className="text-right font-mono font-medium">
                      ₹{item.line_total?.toLocaleString()}
                    </TableCell>
                    <TableCell className="text-center">
                      {isHighRisk ? (
                        <Badge className="bg-destructive/10 text-destructive border border-destructive/30">
                          High
                        </Badge>
                      ) : isMediumRisk ? (
                        <Badge className="bg-warning/10 text-warning border border-warning/30">
                          Medium
                        </Badge>
                      ) : (
                        <CheckCircle2 className="h-5 w-5 text-success mx-auto" />
                      )}
                    </TableCell>
                  </motion.tr>
                );
              })}
            </TableBody>
          </Table>
        </motion.div>

        {/* Summary */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="grid grid-cols-3 gap-4"
        >
          <div className="bg-card rounded-xl border border-border p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-primary/10">
                <DollarSign className="h-5 w-5 text-primary" />
              </div>
              <span className="text-sm text-muted-foreground">Material Price</span>
            </div>
            <p className="text-2xl font-bold font-mono">
              ₹{materialTotal.toLocaleString()}
            </p>
          </div>

          <div className="bg-card rounded-xl border border-border p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-info/10">
                <Calculator className="h-5 w-5 text-info" />
              </div>
              <span className="text-sm text-muted-foreground">Testing & Services</span>
            </div>
            <p className="text-2xl font-bold font-mono">
              ₹{testingTotal.toLocaleString()}
            </p>
          </div>

          <div className="bg-gradient-to-br from-primary/20 to-primary/5 rounded-xl border border-primary/30 p-6">
            <div className="flex items-center gap-3 mb-2">
              <div className="p-2 rounded-lg bg-primary">
                <FileText className="h-5 w-5 text-primary-foreground" />
              </div>
              <span className="text-sm text-primary">Grand Total Bid Price</span>
            </div>
            <p className="text-3xl font-bold font-mono text-primary">
              ₹{grandTotal.toLocaleString()}
            </p>
            {pricingData?.totals_rationale && (
              <Button
                variant="outline"
                size="sm"
                className="mt-3 w-full"
                onClick={() => setShowRationale(true)}
              >
                <Sparkles className="h-3 w-3 mr-2" />
                View Valuation Rationale
              </Button>
            )}
          </div>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex items-center justify-between pt-4"
        >
          <p className="text-sm text-muted-foreground">
            Pricing auto-generated from Product & Pricing Repository
          </p>
          <div className="flex gap-3">
            <Button variant="outline">
              <Calculator className="h-4 w-4 mr-2" />
              Recalculate
            </Button>
            <Button 
              variant="outline"
              onClick={() => setShowViabilityDialog(true)}
              className="bg-gradient-to-r from-blue-600/10 to-purple-600/10 hover:from-blue-600/20 hover:to-purple-600/20 border-blue-500/50"
            >
              <Target className="h-4 w-4 mr-2" />
              Evaluate Viability
            </Button>
            <Button>
              <CheckCircle2 className="h-4 w-4 mr-2" />
              Finalize Pricing & Proceed
            </Button>
          </div>
        </motion.div>
      </div>

      {/* XAI Pricing Rationale Dialog - Individual Item */}
      <Dialog open={showRationale && !!selectedItemRationale} onOpenChange={(open) => {
        if (!open) setSelectedItemRationale("");
        setShowRationale(open);
      }}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Info className="h-5 w-5 text-primary" />
              Pricing Justification
            </DialogTitle>
          </DialogHeader>
          <div className="bg-info/5 border border-info/30 rounded-lg p-4">
            <div className="text-sm text-foreground leading-relaxed whitespace-pre-line space-y-2">
              {selectedItemRationale?.split('**').map((part: string, idx: number) => (
                idx % 2 === 1 ? <strong key={idx} className="text-primary">{part}</strong> : <span key={idx}>{part}</span>
              ))}
            </div>
          </div>
          <div className="border-t pt-4">
            <p className="text-xs text-muted-foreground">
              Pricing is retrieved from the validated Product & Pricing Repository. Testing costs are calculated 
              based on mandatory compliance tests from the Test Data catalog.
            </p>
          </div>
        </DialogContent>
      </Dialog>

      {/* XAI Totals Rationale Dialog */}
      <Dialog open={showRationale && !selectedItemRationale} onOpenChange={setShowRationale}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Explainable AI: Valuation Rationale
            </DialogTitle>
          </DialogHeader>
          {pricingData?.totals_rationale && (
            <div className="space-y-6">
              {/* Material Total Justification */}
              <div className="bg-primary/5 border border-primary/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <DollarSign className="h-4 w-4 text-primary" />
                  <h3 className="font-semibold text-primary">Material Cost Justification</h3>
                </div>
                <div className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line space-y-2">
                  {pricingData.totals_rationale.material_total_justification?.split('**').map((part: string, idx: number) => (
                    idx % 2 === 1 ? <strong key={idx} className="text-foreground">{part}</strong> : <span key={idx}>{part}</span>
                  ))}
                </div>
              </div>

              {/* Testing Total Justification */}
              <div className="bg-info/5 border border-info/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Calculator className="h-4 w-4 text-info" />
                  <h3 className="font-semibold text-info">Testing Cost Justification</h3>
                </div>
                <div className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line space-y-2">
                  {pricingData.totals_rationale.testing_total_justification?.split('**').map((part: string, idx: number) => (
                    idx % 2 === 1 ? <strong key={idx} className="text-foreground">{part}</strong> : <span key={idx}>{part}</span>
                  ))}
                </div>
              </div>

              {/* Grand Total Justification */}
              <div className="bg-success/5 border border-success/30 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <FileText className="h-4 w-4 text-success" />
                  <h3 className="font-semibold text-success">Grand Total Justification</h3>
                </div>
                <div className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line space-y-2">
                  {pricingData.totals_rationale.grand_total_justification?.split('**').map((part: string, idx: number) => (
                    idx % 2 === 1 ? <strong key={idx} className="text-foreground">{part}</strong> : <span key={idx}>{part}</span>
                  ))}
                </div>
              </div>

              {/* Footer Note */}
              <div className="border-t pt-4">
                <p className="text-xs text-muted-foreground">
                  <strong>Audit Note:</strong> These justifications are generated by the Pricing Agent's automated 
                  valuation system. All costs are traceable to the Product Catalog and Test Data collections in MongoDB, 
                  ensuring full transparency and audit compliance. These rationales are for internal dashboard use only 
                  and are NOT included in the final proposal document.
                </p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Technical Viability Gate Dialog */}
      <Dialog open={showViabilityDialog} onOpenChange={setShowViabilityDialog}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Target className="h-5 w-5 text-primary" />
              Technical Viability Assessment
            </DialogTitle>
          </DialogHeader>
          
          <div className="space-y-6">
            {/* Overall Spec Match Display */}
            <div className="text-center p-6 rounded-lg border-2" style={{
              backgroundColor: overallSpecMatch > 70 ? 'rgba(34, 197, 94, 0.1)' : 
                             overallSpecMatch >= 50 ? 'rgba(234, 179, 8, 0.1)' : 
                             'rgba(239, 68, 68, 0.1)',
              borderColor: overallSpecMatch > 70 ? 'rgb(34, 197, 94)' : 
                          overallSpecMatch >= 50 ? 'rgb(234, 179, 8)' : 
                          'rgb(239, 68, 68)'
            }}>
              <div className="flex items-center justify-center gap-2 mb-2">
                {overallSpecMatch > 70 ? (
                  <CheckCircle2 className="h-8 w-8 text-green-600" />
                ) : overallSpecMatch >= 50 ? (
                  <AlertCircle className="h-8 w-8 text-yellow-600" />
                ) : (
                  <XCircle className="h-8 w-8 text-red-600" />
                )}
              </div>
              <h3 className="text-2xl font-bold mb-1" style={{
                color: overallSpecMatch > 70 ? 'rgb(34, 197, 94)' : 
                       overallSpecMatch >= 50 ? 'rgb(234, 179, 8)' : 
                       'rgb(239, 68, 68)'
              }}>
                Overall Spec Match: {overallSpecMatch.toFixed(1)}%
              </h3>
              <p className="text-sm text-muted-foreground">
                Average match score across all {items.length} item(s)
              </p>
            </div>

            {/* Conditional UI based on score */}
            {overallSpecMatch > 70 && (
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-green-600/10 border border-green-600/30">
                  <p className="text-green-700 dark:text-green-400 font-semibold mb-2">
                    ✅ High Viability Detected
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Suggestion: The technical specifications match our capabilities very well. 
                    This RFP has a high probability of success. Consider proceeding to the Final Proposal stage.
                  </p>
                </div>
                <div className="flex gap-3">
                  <Button 
                    className="flex-1 bg-green-600 hover:bg-green-700"
                    onClick={() => setShowViabilityDialog(false)}
                  >
                    <CheckCircle2 className="h-4 w-4 mr-2" />
                    Continue to Proposal
                  </Button>
                  <Button 
                    variant="outline"
                    className="flex-1"
                    onClick={() => {
                      setShowViabilityDialog(false);
                      handleSkipRFP();
                    }}
                    disabled={isSkipping}
                  >
                    {isSkipping ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <SkipForward className="h-4 w-4 mr-2" />}
                    Manual Skip
                  </Button>
                </div>
              </div>
            )}

            {overallSpecMatch >= 50 && overallSpecMatch <= 70 && (
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-yellow-600/10 border border-yellow-600/30">
                  <p className="text-yellow-700 dark:text-yellow-400 font-semibold mb-2">
                    ⚠️ Medium Viability
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Suggestion: The technical match is moderate. Consider saving this RFP to the Repository 
                    for further review, or skip it to find a better opportunity.
                  </p>
                </div>
                <div className="flex gap-3">
                  <Button 
                    variant="outline"
                    className="flex-1"
                    disabled
                    title="Coming Soon"
                  >
                    <FileText className="h-4 w-4 mr-2" />
                    Save & Cycle (Coming Soon)
                  </Button>
                  <Button 
                    variant="destructive"
                    className="flex-1"
                    onClick={() => {
                      setShowViabilityDialog(false);
                      handleSkipRFP();
                    }}
                    disabled={isSkipping}
                  >
                    {isSkipping ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <SkipForward className="h-4 w-4 mr-2" />}
                    Cycle Without Saving
                  </Button>
                </div>
              </div>
            )}

            {overallSpecMatch < 50 && (
              <div className="space-y-4">
                <div className="p-4 rounded-lg bg-red-600/10 border border-red-600/30">
                  <p className="text-red-700 dark:text-red-400 font-semibold mb-2">
                    ❌ Low Viability
                  </p>
                  <p className="text-sm text-muted-foreground">
                    Suggestion: The technical specifications do not match our capabilities well. 
                    It is recommended to discard this RFP and find a better opportunity to maximize your win rate.
                  </p>
                </div>
                <Button 
                  variant="destructive"
                  className="w-full bg-red-600 hover:bg-red-700"
                  onClick={() => {
                    setShowViabilityDialog(false);
                    handleSkipRFP();
                  }}
                  disabled={isSkipping}
                >
                  {isSkipping ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <XCircle className="h-4 w-4 mr-2" />}
                  Discard & Cycle
                </Button>
              </div>
            )}

            {/* Item-level breakdown */}
            <div className="border-t pt-4">
              <p className="text-xs font-semibold text-muted-foreground uppercase mb-3">
                Item-Level Breakdown
              </p>
              <div className="space-y-2 max-h-40 overflow-y-auto">
                {items.map((item, idx) => (
                  <div key={idx} className="flex items-center justify-between text-sm p-2 rounded bg-muted/50">
                    <span className="text-muted-foreground truncate flex-1">
                      {item.item_name || item.sku_code}
                    </span>
                    <Badge 
                      variant="outline"
                      className={cn(
                        "ml-2",
                        item.spec_match_percent > 70 ? "bg-green-600/10 text-green-700 border-green-600/30" :
                        item.spec_match_percent >= 50 ? "bg-yellow-600/10 text-yellow-700 border-yellow-600/30" :
                        "bg-red-600/10 text-red-700 border-red-600/30"
                      )}
                    >
                      {item.spec_match_percent?.toFixed(1)}%
                    </Badge>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </MainLayout>
  );
}
