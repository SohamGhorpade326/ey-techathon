import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
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
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import {
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  AlertTriangle,
  Eye,
  MessageSquare,
  Sparkles,
  Loader2,
} from "lucide-react";
import { cn } from "@/lib/utils";

const API = "http://127.0.0.1:8000";

export default function Analysis() {
  const [expandedRows, setExpandedRows] = useState<number[]>([]);
  const [technicalData, setTechnicalData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [selectedSKUs, setSelectedSKUs] = useState<Record<number, string>>({});
  const [showComparison, setShowComparison] = useState(false);
  const [selectedItemForComparison, setSelectedItemForComparison] = useState<any>(null);
  const [showOverride, setShowOverride] = useState(false);
  const [overrideComment, setOverrideComment] = useState("");
  const [showJustification, setShowJustification] = useState(false);
  const [selectedItemForJustification, setSelectedItemForJustification] = useState<any>(null);

  useEffect(() => {
    fetch(`${API}/technical/output`)
      .then((res) => res.json())
      .then((data) => {
        setTechnicalData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching technical data:", err);
        setLoading(false);
      });
  }, []);

  if (loading) {
    return (
      <MainLayout title="Technical Spec-Matching Workspace">
        <div className="flex h-[60vh] items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </MainLayout>
    );
  }

  if (!technicalData || !technicalData.items || technicalData.items.length === 0) {
    return (
      <MainLayout title="Technical Spec-Matching Workspace">
        <div className="text-center py-20">
          <p className="text-muted-foreground">No technical analysis data available. Please run the Sales Agent first.</p>
        </div>
      </MainLayout>
    );
  }

  const items = technicalData.items || [];
  const avgMatch = items.length > 0 
    ? (items.reduce((sum: number, item: any) => sum + (item.spec_match_percent || 0), 0) / items.length).toFixed(0)
    : 0;

  const toggleRow = (id: number) => {
    setExpandedRows((prev) =>
      prev.includes(id) ? prev.filter((r) => r !== id) : [...prev, id]
    );
  };

  const selectSKU = (itemId: number, sku: string) => {
    setSelectedSKUs((prev) => ({ ...prev, [itemId]: sku }));
  };

  const showComparisonMatrix = (item: any) => {
    setSelectedItemForComparison(item);
    setShowComparison(true);
  };

  const showJustificationDialog = (item: any) => {
    setSelectedItemForJustification(item);
    setShowJustification(true);
  };

  const getMatchColor = (match: number) => {
    if (match >= 90) return "bg-success/10 text-success border-success/30";
    if (match >= 70) return "bg-warning/10 text-warning border-warning/30";
    return "bg-destructive/10 text-destructive border-destructive/30";
  };

  const getMatchBg = (match: number) => {
    if (match >= 90) return "bg-success";
    if (match >= 70) return "bg-warning";
    return "bg-destructive";
  };

  const getStatusIcon = (status: string) => {
    if (status === "Matched" || status === "Warning") {
      return <CheckCircle2 className="h-5 w-5 text-success mx-auto" />;
    }
    return <AlertTriangle className="h-5 w-5 text-warning mx-auto" />;
  };

  return (
    <MainLayout
      title="Technical Spec-Matching Workspace"
      breadcrumbs={[
        { name: "Dashboard", href: "/dashboard" },
        { name: "AI Analysis Workspace" },
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
              RFP {technicalData.rfp_id} - {technicalData.domain}
            </p>
            <div className="flex items-center gap-4 mt-2">
              <Badge variant="outline" className="bg-success/10 text-success border-success/30">
                <Sparkles className="h-3 w-3 mr-1" />
                Technical Agent {technicalData.status}
              </Badge>
              <span className="text-sm text-muted-foreground">
                {items.length} items matched • Avg Spec-Match: {avgMatch}%
              </span>
            </div>
          </div>
        </motion.div>

        {/* Info Banner */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="bg-info/10 border border-info/30 rounded-lg p-4"
        >
          <p className="text-sm text-info">
            <strong>Automation Insight:</strong> Technical SKU matching is the most time-consuming manual task — now automated. Review and confirm matches below.
          </p>
        </motion.div>

        {/* Spec Matching Table */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="bg-card rounded-xl border border-border overflow-hidden"
        >
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead className="w-8"></TableHead>
                <TableHead>Item Name</TableHead>
                <TableHead>Required Technical Specs</TableHead>
                <TableHead>Best Match SKU</TableHead>
                <TableHead className="text-center">Spec-Match %</TableHead>
                <TableHead className="text-center">Status</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items.map((item: any, index: number) => (
                <>
                  <TableRow
                    key={index}
                    className={cn(
                      "cursor-pointer transition-colors",
                      expandedRows.includes(index) && "bg-muted/30"
                    )}
                    onClick={() => toggleRow(index)}
                  >
                    <TableCell>
                      <motion.div
                        animate={{ rotate: expandedRows.includes(index) ? 90 : 0 }}
                        transition={{ duration: 0.2 }}
                      >
                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                      </motion.div>
                    </TableCell>
                    <TableCell className="font-medium">{item.item_name}</TableCell>
                    <TableCell className="text-sm text-muted-foreground max-w-xs truncate">
                      {item.required_specs_text}
                    </TableCell>
                    <TableCell className="font-mono text-sm">
                      {selectedSKUs[index] || item.best_match_sku}
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge className={cn("border", getMatchColor(item.spec_match_percent))}>
                        {item.spec_match_percent?.toFixed(0)}%
                      </Badge>
                    </TableCell>
                    <TableCell className="text-center">
                      {getStatusIcon(item.status)}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            showJustificationDialog(item);
                          }}
                          title="View AI Justification"
                        >
                          <Sparkles className="h-4 w-4 text-primary" />
                        </Button>
                        {item.comparison_matrix && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={(e) => {
                              e.stopPropagation();
                              showComparisonMatrix(item);
                            }}
                          >
                            <Eye className="h-4 w-4 mr-1" />
                            Compare
                          </Button>
                        )}
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            setShowOverride(true);
                          }}
                        >
                          <MessageSquare className="h-4 w-4" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                  <AnimatePresence>
                    {expandedRows.includes(index) && (
                      <TableRow>
                        <TableCell colSpan={7} className="p-0">
                          <motion.div
                            initial={{ height: 0, opacity: 0 }}
                            animate={{ height: "auto", opacity: 1 }}
                            exit={{ height: 0, opacity: 0 }}
                            transition={{ duration: 0.2 }}
                            className="overflow-hidden"
                          >
                            <div className="p-4 bg-muted/20 space-y-3">
                              <p className="text-sm font-medium">Alternative SKU Matches:</p>
                              <div className="grid grid-cols-3 gap-3">
                                {item.top_3_alternatives?.map((match: any, idx: number) => (
                                  <motion.div
                                    key={match.sku_code}
                                    initial={{ opacity: 0, y: 10 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: idx * 0.1 }}
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      selectSKU(index, match.sku_code);
                                    }}
                                    className={cn(
                                      "p-3 rounded-lg border cursor-pointer transition-all",
                                      selectedSKUs[index] === match.sku_code || (!selectedSKUs[index] && idx === 0)
                                        ? "border-primary bg-primary/5"
                                        : "border-border bg-card hover:border-primary/50"
                                    )}
                                  >
                                    <div className="flex items-center justify-between mb-2">
                                      <span className="font-mono text-sm">{match.sku_code}</span>
                                      <Badge className={cn("text-xs border", getMatchColor(match.spec_match_percent))}>
                                        {match.spec_match_percent?.toFixed(0)}%
                                      </Badge>
                                    </div>
                                    <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
                                      <div
                                        className={cn("h-full rounded-full", getMatchBg(match.spec_match_percent))}
                                        style={{ width: `${match.spec_match_percent}%` }}
                                      />
                                    </div>
                                    <p className="text-xs text-muted-foreground mt-2 mb-1">
                                      ₹{match.price_per_unit?.toLocaleString()}/unit
                                    </p>
                                    <p className="text-xs text-muted-foreground line-clamp-2">
                                      {match.sku_description}
                                    </p>
                                  </motion.div>
                                ))}
                              </div>
                            </div>
                          </motion.div>
                        </TableCell>
                      </TableRow>
                    )}
                  </AnimatePresence>
                </>
              ))}
            </TableBody>
          </Table>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="flex items-center justify-between"
        >
          <p className="text-sm text-muted-foreground">
            {Object.keys(selectedSKUs).length} of {items.length} items confirmed
          </p>
          <div className="flex gap-3">
            <Button variant="outline">Save Draft</Button>
            <Button>
              <CheckCircle2 className="h-4 w-4 mr-2" />
              Confirm Selected SKUs
            </Button>
          </div>
        </motion.div>
      </div>
      
      {/* XAI Justification Dialog */}
      <Dialog open={showJustification} onOpenChange={setShowJustification}>
        <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <Sparkles className="h-5 w-5 text-primary" />
              Explainable AI: Match Justification
            </DialogTitle>
          </DialogHeader>
          {selectedItemForJustification && (
            <div className="space-y-6">
              {/* Primary Match Justification */}
              <div className="bg-success/5 border border-success/30 rounded-lg p-4">
                <h3 className="font-semibold text-success mb-2 flex items-center gap-2">
                  <CheckCircle2 className="h-4 w-4" />
                  Primary Match: {selectedItemForJustification.best_match_sku}
                </h3>
                <div className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line space-y-2">
                  {selectedItemForJustification.match_justification?.split('**').map((part: string, idx: number) => (
                    idx % 2 === 1 ? <strong key={idx} className="text-foreground">{part}</strong> : <span key={idx}>{part}</span>
                  ))}
                </div>
              </div>

              {/* Alternative Matches with Delta Explanations */}
              {selectedItemForJustification.top_3_skus && selectedItemForJustification.top_3_skus.length > 1 && (
                <div>
                  <h3 className="font-semibold text-foreground mb-3">Alternative Matches</h3>
                  <div className="space-y-3">
                    {selectedItemForJustification.top_3_skus.slice(1).map((alt: any, idx: number) => (
                      <div key={alt.sku_code} className="bg-muted/30 border border-border rounded-lg p-4">
                        <div className="flex items-start justify-between mb-2">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-sm font-medium">{alt.sku_code}</span>
                            <Badge className={cn("text-xs border", getMatchColor(alt.spec_match_percent))}>
                              {alt.spec_match_percent?.toFixed(0)}%
                            </Badge>
                          </div>
                          <Badge variant="outline" className="text-xs">
                            Alternative #{idx + 2}
                          </Badge>
                        </div>
                        <div className="text-sm text-muted-foreground leading-relaxed whitespace-pre-line">
                          {alt.delta_explanation?.split('**').map((part: string, i: number) => (
                            i % 2 === 1 ? <strong key={i} className="text-foreground">{part}</strong> : <span key={i}>{part}</span>
                          )) || "Alternative SKU with lower match score."}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Technical Details */}
              <div className="border-t pt-4">
                <p className="text-xs text-muted-foreground">
                  <strong>Note:</strong> These justifications are generated by the Technical Agent's Parameter Rule Engine 
                  and SBERT semantic matching system. Vector similarity scores confirm the alignment between RFP requirements 
                  and catalog specifications.
                </p>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
      
      {/* Comparison Dialog */}
      <Dialog open={showComparison} onOpenChange={setShowComparison}>
        <DialogContent className="max-w-5xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>Detailed Spec Comparison - {selectedItemForComparison?.item_name}</DialogTitle>
          </DialogHeader>
          {selectedItemForComparison?.comparison_matrix && (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Spec Parameter</TableHead>
                  <TableHead>RFP Requirement</TableHead>
                  {selectedItemForComparison.comparison_matrix.comparison_rows[0]?.sku_1 && (
                    <TableHead>
                      SKU #1 ({selectedItemForComparison.top_3_alternatives?.[0]?.spec_match_percent?.toFixed(0)}%)
                    </TableHead>
                  )}
                  {selectedItemForComparison.comparison_matrix.comparison_rows[0]?.sku_2 && (
                    <TableHead>
                      SKU #2 ({selectedItemForComparison.top_3_alternatives?.[1]?.spec_match_percent?.toFixed(0)}%)
                    </TableHead>
                  )}
                  {selectedItemForComparison.comparison_matrix.comparison_rows[0]?.sku_3 && (
                    <TableHead>
                      SKU #3 ({selectedItemForComparison.top_3_alternatives?.[2]?.spec_match_percent?.toFixed(0)}%)
                    </TableHead>
                  )}
                </TableRow>
              </TableHeader>
              <TableBody>
                {selectedItemForComparison.comparison_matrix.comparison_rows.map((row: any, idx: number) => (
                  <TableRow key={idx}>
                    <TableCell className="font-medium">{row.parameter}</TableCell>
                    <TableCell className="bg-primary/5">{row.rfp_requirement}</TableCell>
                    {row.sku_1 && (
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{row.sku_1.sku_code}</span>
                          {row.sku_1.match ? (
                            <CheckCircle2 className="h-4 w-4 text-success" />
                          ) : (
                            <AlertTriangle className="h-4 w-4 text-warning" />
                          )}
                        </div>
                      </TableCell>
                    )}
                    {row.sku_2 && (
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{row.sku_2.sku_code}</span>
                          {row.sku_2.match ? (
                            <CheckCircle2 className="h-4 w-4 text-success" />
                          ) : (
                            <AlertTriangle className="h-4 w-4 text-warning" />
                          )}
                        </div>
                      </TableCell>
                    )}
                    {row.sku_3 && (
                      <TableCell>
                        <div className="flex items-center gap-2">
                          <span className="text-sm">{row.sku_3.sku_code}</span>
                          {row.sku_3.match ? (
                            <CheckCircle2 className="h-4 w-4 text-success" />
                          ) : (
                            <AlertTriangle className="h-4 w-4 text-warning" />
                          )}
                        </div>
                      </TableCell>
                    )}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </DialogContent>
      </Dialog>

      {/* Override Dialog */}
      <Dialog open={showOverride} onOpenChange={setShowOverride}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Add Override Comment</DialogTitle>
          </DialogHeader>
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              Provide a reason for overriding the suggested SKU match (e.g., "New MTO SKU needed").
            </p>
            <Textarea
              value={overrideComment}
              onChange={(e) => setOverrideComment(e.target.value)}
              placeholder="Enter your comment..."
              rows={4}
            />
            <div className="flex justify-end gap-3">
              <Button variant="outline" onClick={() => setShowOverride(false)}>
                Cancel
              </Button>
              <Button onClick={() => setShowOverride(false)}>
                Save Override
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </MainLayout>
  );
}
