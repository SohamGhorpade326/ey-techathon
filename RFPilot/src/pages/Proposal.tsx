import { useState, useEffect, useRef } from "react";
import { motion } from "framer-motion";
import { MainLayout } from "@/components/layout/MainLayout";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { useToast } from "@/hooks/use-toast";
import ReactQuill from "react-quill";
import "react-quill/dist/quill.snow.css";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  CheckCircle2,
  Clock,
  Download,
  FileText,
  History,
  Save,
  Send,
  Sparkles,
  Loader2,
  Edit3,
  RotateCcw,
  X,
  Paperclip,
} from "lucide-react";
import jsPDF from "jspdf";
import { Document, Packer, Paragraph, TextRun } from "docx";

const API = "http://127.0.0.1:8000";

const versions = [
  { id: 1, version: "v1.0", date: "2024-01-15 10:30", author: "System", status: "Draft" },
  { id: 2, version: "v1.1", date: "2024-01-15 14:45", author: "Sales Lead", status: "Draft" },
  { id: 3, version: "v1.2", date: "2024-01-16 09:15", author: "Sales Lead", status: "Review" },
];

export default function Proposal() {
  const [coverContent, setCoverContent] = useState("");
  const [editableBidContent, setEditableBidContent] = useState("");
  const [complianceItems, setComplianceItems] = useState<any[]>([]);
  const [pricingTotals, setPricingTotals] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [discount, setDiscount] = useState(2);
  const [discountApplied, setDiscountApplied] = useState(false);
  const [savingBid, setSavingBid] = useState(false);
  const [generatingCoverLetter, setGeneratingCoverLetter] = useState(false);
  const [coverLetterMode, setCoverLetterMode] = useState<"initial" | "ai" | "manual">("initial");
  
  // Gmail modal states
  const [showGmailModal, setShowGmailModal] = useState(false);
  const [emailData, setEmailData] = useState({
    to: "",
    subject: "",
    body: ""
  });
  const [attachments, setAttachments] = useState<File[]>([]);
  const [sendingEmail, setSendingEmail] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const { toast } = useToast();

  useEffect(() => {
    Promise.all([
      fetch(`${API}/proposal`).then(res => res.json()),
      fetch(`${API}/compliance`).then(res => res.json()),
      fetch(`${API}/pricing/totals`).then(res => res.json())
    ])
      .then(([proposalData, complianceData, totalsData]) => {
        // Format proposal content professionally
        const rawProposal = proposalData.proposal || "";
        const formattedProposal = formatProposalText(rawProposal);
        setCoverContent(formattedProposal);
        setEditableBidContent(formattedProposal); // Initialize editable content
        setComplianceItems(complianceData);
        setPricingTotals(totalsData);
        setLoading(false);
      })
      .catch(err => {
        console.error("Error fetching proposal data:", err);
        setLoading(false);
      });
  }, []);

  const formatProposalText = (text: string): string => {
    // Split into paragraphs and format professionally
    return text
      .split('\n\n')
      .map(para => para.trim())
      .filter(para => para.length > 0)
      .join('\n\n');
  };

  const handleApplyDiscount = () => {
    setDiscountApplied(true);
  };

  const handleSaveBidChanges = async () => {
    setSavingBid(true);
    try {
      const response = await fetch(`${API}/proposal/edit`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          updated_bid: editableBidContent
        })
      });

      const result = await response.json();

      if (result.status === "success") {
        toast({
          title: "Changes Saved",
          description: "Your proposal edits have been saved successfully.",
          duration: 3000,
        });
      } else {
        throw new Error(result.message || "Failed to save changes");
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to save changes. Please try again.",
        variant: "destructive",
        duration: 3000,
      });
    } finally {
      setSavingBid(false);
    }
  };

  const handleGenerateAICoverLetter = async () => {
    setGeneratingCoverLetter(true);
    try {
      const response = await fetch(`${API}/proposal/generate-cover-letter`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          buyer: "State Electricity Board",
          rfp_title: "Power Cable Supply",
          rfp_id: "RFP/SEB/2024/1023"
        })
      });

      const result = await response.json();

      if (result.status === "success") {
        setCoverContent(result.cover_letter);
        setCoverLetterMode("ai");
        toast({
          title: "Cover Letter Generated",
          description: "AI-powered cover letter has been created. You can edit it before saving.",
          duration: 3000,
        });
      } else {
        throw new Error(result.message || "Failed to generate cover letter");
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to generate cover letter. Please try again.",
        variant: "destructive",
        duration: 3000,
      });
    } finally {
      setGeneratingCoverLetter(false);
    }
  };

  const handleManualCoverLetter = () => {
    setCoverContent("");
    setCoverLetterMode("manual");
  };

  const handleResetCoverLetter = () => {
    setCoverContent("");
    setCoverLetterMode("initial");
  };

  const handleMarkReadyToSubmit = () => {
    // Determine recipient based on RFP
    const rfpTitle = "Painting and repair work of National Housing Bank's flats located at Mumbai";
    let recipient = "";
    
    if (rfpTitle.toLowerCase().includes("nhb") || rfpTitle.toLowerCase().includes("housing bank")) {
      recipient = "romum@nhb.org.in";
    } else if (rfpTitle.toLowerCase().includes("sac") || rfpTitle.toLowerCase().includes("isro")) {
      recipient = "stores_receipt@sac.isro.gov.in";
    }
    
    // Pre-fill email data with ONLY the cover letter (not full bid)
    const emailBody = coverContent || "Please generate a cover letter first.";
    setEmailData({
      to: recipient,
      subject: `Proposal Submission - ${rfpTitle}`,
      body: emailBody
    });
    setShowGmailModal(true);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setAttachments(prev => [...prev, ...files]);
    }
  };

  const handleRemoveAttachment = (index: number) => {
    setAttachments(prev => prev.filter((_, i) => i !== index));
  };

  const handleSendEmail = async () => {
    if (!emailData.to || !emailData.subject) {
      toast({
        title: "Missing Information",
        description: "Please fill in recipient and subject fields.",
        variant: "destructive",
        duration: 3000,
      });
      return;
    }

    setSendingEmail(true);
    try {
      const formData = new FormData();
      formData.append("to_email", emailData.to);
      formData.append("subject", emailData.subject);
      formData.append("html_content", emailData.body);
      
      // Append all attachments
      attachments.forEach((file) => {
        formData.append("attachments", file);
      });

      const response = await fetch(`${API}/proposal/send-email`, {
        method: "POST",
        body: formData
      });

      const result = await response.json();

      if (result.status === "success") {
        toast({
          title: "Email Sent Successfully",
          description: `Bid submitted to ${emailData.to} via SendGrid.`,
          duration: 3000,
        });
        setShowGmailModal(false);
        setAttachments([]);
      } else {
        throw new Error(result.message || "Failed to send email");
      }
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to send email. Please try again.",
        variant: "destructive",
        duration: 3000,
      });
    } finally {
      setSendingEmail(false);
    }
  };

  const calculateDiscountedTotal = () => {
    if (!pricingTotals || !discountApplied) return pricingTotals?.grand_total || 0;
    const discountAmount = (pricingTotals.grand_total * discount) / 100;
    return pricingTotals.grand_total - discountAmount;
  };

  const getDiscountAmount = () => {
    if (!pricingTotals || !discountApplied) return 0;
    return (pricingTotals.grand_total * discount) / 100;
  };

  const handleExportPDF = () => {
    try {
      const pdf = new jsPDF("p", "mm", "a4");
      
      // Use the current editable content instead of coverContent
      const contentToExport = editableBidContent || coverContent;
      
      // Text sanitization helper
      const cleanText = (t: string) =>
        t
          .replace(/₹/g, "Rs.")  // Replace rupee symbol with ASCII equivalent
          .replace(/['']/g, "'")
          .replace(/[""]/g, '"')
          .replace(/[–—]/g, "-")
          .replace(/[\u0080-\uFFFF]/g, (char) => {
            // Replace any remaining non-ASCII characters
            const code = char.charCodeAt(0);
            if (code > 127) return "";
            return char;
          })
          .normalize("NFKD")
          .replace(/[\u0300-\u036f]/g, ""); // Remove combining diacritical marks
      
      // Force consistent font globally
      pdf.setFont("helvetica", "normal");
      
      // Title section
      pdf.setFontSize(18);
      pdf.setFont("helvetica", "bold");
      pdf.text(cleanText("BID RESPONSE"), 105, 20, { align: "center" });
      
      pdf.setFontSize(11);
      pdf.setFont("helvetica", "normal");
      pdf.text(cleanText("RFP Reference: RFP/SEB/2024/1023"), 105, 28, { align: "center" });
      
      
      pdf.line(15, 40, 195, 40);
      
      // Content
      pdf.setFont("helvetica", "normal");
      pdf.setFontSize(10);
      let yPosition = 50;
      const lineHeight = 6;
      const pageHeight = pdf.internal.pageSize.getHeight();
      const margin = 15;
      const maxWidth = 180;
      
      const paragraphs = contentToExport.split('\n\n');
      
      paragraphs.forEach((paragraph) => {
        const cleanedParagraph = cleanText(paragraph);
        const lines = pdf.splitTextToSize(cleanedParagraph, maxWidth);
        
        lines.forEach((line: string) => {
          if (yPosition > pageHeight - 20) {
            pdf.addPage();
            pdf.setFont("helvetica", "normal");
            pdf.setFontSize(10);
            yPosition = 20;
          }
          pdf.text(line, margin, yPosition);
          yPosition += lineHeight;
        });
        
        yPosition += 4; // Extra spacing between paragraphs
      });
      
      pdf.save("proposal.pdf");
    } catch (error) {
      console.error("Error generating PDF:", error);
    }
  };

  const handleExportWord = async () => {
    try {
      // Use the current editable content instead of coverContent
      const contentToExport = editableBidContent || coverContent;
      const paragraphs = contentToExport.split('\n\n').filter(p => p.trim().length > 0);
      
      const doc = new Document({
        sections: [{
          properties: {},
          children: [
            // Title
            new Paragraph({
              children: [new TextRun({
                text: "BID RESPONSE",
                bold: true,
                size: 32,
              })],
              alignment: "center" as any,
              spacing: { after: 200 },
            }),
            // Subtitle
            new Paragraph({
              children: [new TextRun({
                text: "RFP Reference: RFP/SEB/2024/1023",
                size: 22,
              })],
              alignment: "center" as any,
              spacing: { after: 100 },
            }),
            new Paragraph({
              children: [new TextRun({
                text: "State Electricity Board - Power Cable Supply",
                size: 22,
              })],
              alignment: "center" as any,
              spacing: { after: 400 },
              border: {
                bottom: {
                  color: "000000",
                  space: 1,
                  style: "single" as any,
                  size: 6,
                },
              },
            }),
            // Content paragraphs
            ...paragraphs.map(para => 
              new Paragraph({
                children: [new TextRun({
                  text: para,
                  size: 22,
                })],
                spacing: { 
                  after: 240,
                  line: 360,
                },
              })
            ),
          ],
        }],
      });
      
      const blob = await Packer.toBlob(doc);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = "proposal.docx";
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Error generating Word document:", error);
    }
  };

  if (loading) {
    return (
      <MainLayout title="Proposal Builder">
        <div className="flex h-[60vh] items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-primary" />
        </div>
      </MainLayout>
    );
  }

  if (!pricingTotals || !complianceItems.length) {
    return (
      <MainLayout title="Proposal Builder">
        <div className="text-center py-20">
          <p className="text-muted-foreground">No proposal data available. Please run the complete pipeline first.</p>
        </div>
      </MainLayout>
    );
  }

  return (
    <MainLayout
      title="Proposal Builder"
      breadcrumbs={[
        { name: "Dashboard", href: "/dashboard" },
        { name: "Proposal Builder" },
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
              RFP #1023 - State Electricity Board - Power Cable Supply
            </p>
            <div className="flex items-center gap-4 mt-2">
              <Badge variant="outline" className="bg-agent-master/10 text-agent-master border-agent-master/30">
                <Sparkles className="h-3 w-3 mr-1" />
                Master Agent Consolidated
              </Badge>
            </div>
          </div>
          <div className="flex gap-3">
            <Select defaultValue="v1.2">
              <SelectTrigger className="w-32">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {versions.map((v) => (
                  <SelectItem key={v.id} value={v.version}>
                    {v.version}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="outline">
              <History className="h-4 w-4 mr-2" />
              History
            </Button>
          </div>
        </motion.div>

        {/* Status Line */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex items-center gap-6 p-4 bg-card rounded-xl border border-border"
        >
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-success" />
            <span className="text-sm">Technical aligned</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-success" />
            <span className="text-sm">Pricing aligned</span>
          </div>
          <div className="h-4 w-px bg-border" />
          <div className="flex items-center gap-2">
            <Clock className="h-5 w-5 text-warning" />
            <span className="text-sm">Submission pending</span>
          </div>
          <div className="ml-auto">
            <Badge className="bg-warning/10 text-warning border border-warning/30">
              Due in 8 days
            </Badge>
          </div>
        </motion.div>

        {/* Proposal Content */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <Tabs defaultValue="cover" className="space-y-4">
            <TabsList className="bg-muted/50">
              <TabsTrigger value="cover">Cover Letter</TabsTrigger>
              <TabsTrigger value="compliance">Product Compliance</TabsTrigger>
              <TabsTrigger value="pricing">Pricing Summary</TabsTrigger>
              <TabsTrigger value="preview">Full Preview</TabsTrigger>
            </TabsList>

            <TabsContent value="cover" className="space-y-4">
              <div className="bg-card rounded-xl border border-border p-6">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Cover Letter</h3>
                  <span className="text-xs text-muted-foreground">
                    AI-powered or manually written
                  </span>
                </div>
                
                {coverLetterMode === "initial" && (
                  <div className="flex flex-col items-center justify-center py-16 space-y-4">
                    <p className="text-muted-foreground text-center mb-4">
                      Choose how you'd like to create your cover letter
                    </p>
                    <div className="flex gap-4">
                      <Button 
                        onClick={handleGenerateAICoverLetter}
                        disabled={generatingCoverLetter}
                        className="gap-2"
                      >
                        {generatingCoverLetter ? (
                          <>
                            <Loader2 className="h-4 w-4 animate-spin" />
                            Generating...
                          </>
                        ) : (
                          <>
                            <Sparkles className="h-4 w-4" />
                            Generate with AI
                          </>
                        )}
                      </Button>
                      <Button 
                        variant="outline"
                        onClick={handleManualCoverLetter}
                        className="gap-2"
                      >
                        <Edit3 className="h-4 w-4" />
                        Enter Manually
                      </Button>
                    </div>
                  </div>
                )}

                {(coverLetterMode === "ai" || coverLetterMode === "manual") && (
                  <div className="space-y-4">
                    <Textarea
                      value={coverContent}
                      onChange={(e) => setCoverContent(e.target.value)}
                      className="min-h-[400px] font-mono text-sm"
                      placeholder={coverLetterMode === "manual" ? "Enter your cover letter here..." : ""}
                    />
                    <div className="flex items-center justify-between">
                      <Button
                        variant="outline"
                        onClick={handleResetCoverLetter}
                        className="gap-2"
                      >
                        <RotateCcw className="h-4 w-4" />
                        Reset
                      </Button>
                      <Button
                        onClick={() => {
                          toast({
                            title: "Cover Letter Saved",
                            description: "Your cover letter has been saved successfully.",
                            duration: 3000,
                          });
                        }}
                        className="gap-2"
                      >
                        <Save className="h-4 w-4" />
                        Save Cover Letter
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </TabsContent>

            <TabsContent value="compliance" className="space-y-4">
              <div className="bg-card rounded-xl border border-border overflow-hidden">
                <div className="p-4 border-b border-border">
                  <h3 className="font-semibold">Product Compliance Table</h3>
                  <p className="text-sm text-muted-foreground">
                    Auto-generated from Technical Agent analysis
                  </p>
                </div>
                <Table>
                  <TableHeader>
                    <TableRow className="bg-muted/50">
                      <TableHead>RFP Item</TableHead>
                      <TableHead>Unit Price</TableHead>
                      <TableHead className="text-center">Compliance</TableHead>
                      <TableHead className="text-center">Spec-Match</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {complianceItems.map((item, index) => (
                      <TableRow key={index}>
                        <TableCell className="font-medium">{item.rfp_item}</TableCell>
                        <TableCell className="font-mono text-sm">₹{item.unit_price?.toLocaleString()}/unit</TableCell>
                        <TableCell className="text-center">
                          <Badge
                            className={
                              item.compliance === "Full"
                                ? "bg-success/10 text-success border border-success/30"
                                : "bg-warning/10 text-warning border border-warning/30"
                            }
                          >
                            {item.compliance}
                          </Badge>
                        </TableCell>
                        <TableCell className="text-center">
                          <Badge
                            className={
                              item.spec_match_percent >= 90
                                ? "bg-success/10 text-success border border-success/30"
                                : item.spec_match_percent >= 70
                                ? "bg-warning/10 text-warning border border-warning/30"
                                : "bg-destructive/10 text-destructive border border-destructive/30"
                            }
                          >
                            {item.spec_match_percent?.toFixed(1)}%
                          </Badge>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </TabsContent>

            <TabsContent value="pricing" className="space-y-4">
              <div className="bg-card rounded-xl border border-border p-6">
                <div className="mb-4">
                  <h3 className="font-semibold">Consolidated Pricing</h3>
                  <p className="text-sm text-muted-foreground">
                    From Pricing Agent calculation
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-6 mb-6">
                  <div className="p-4 bg-muted/30 rounded-lg">
                    <p className="text-sm text-muted-foreground">Material Cost</p>
                    <p className="text-2xl font-bold font-mono mt-1">₹{pricingTotals.material_total?.toLocaleString()}</p>
                  </div>
                  <div className="p-4 bg-muted/30 rounded-lg">
                    <p className="text-sm text-muted-foreground">Testing & Services</p>
                    <p className="text-2xl font-bold font-mono mt-1">₹{pricingTotals.testing_total?.toLocaleString()}</p>
                  </div>
                  <div className="p-4 bg-muted/30 rounded-lg border border-border">
                    <p className="text-sm text-muted-foreground">Subtotal</p>
                    <p className="text-2xl font-bold font-mono mt-1">₹{pricingTotals.grand_total?.toLocaleString()}</p>
                  </div>
                </div>

                {/* Discount Section */}
                <div className="border-t border-border pt-6 space-y-4">
                  <div className="flex items-center gap-4">
                    <div className="flex-1">
                      <label className="text-sm font-medium mb-2 block">Discount Percentage</label>
                      <div className="flex items-center gap-3">
                        <input
                          type="number"
                          min="0"
                          max="100"
                          step="0.1"
                          value={discount}
                          onChange={(e) => setDiscount(parseFloat(e.target.value) || 0)}
                          className="w-32 px-3 py-2 border border-border rounded-lg bg-background text-foreground focus:outline-none focus:ring-2 focus:ring-primary"
                        />
                        <span className="text-muted-foreground">%</span>
                        <Button 
                          onClick={handleApplyDiscount}
                          className="ml-2"
                        >
                          Apply Discount
                        </Button>
                      </div>
                    </div>
                  </div>

                  {discountApplied && (
                    <div className="space-y-3 pt-4 border-t border-border">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Discount ({discount}%)</span>
                        <span className="font-mono text-destructive">-₹{getDiscountAmount().toLocaleString()}</span>
                      </div>
                      <div className="p-4 bg-primary/10 rounded-lg border border-primary/30">
                        <div className="flex items-center justify-between">
                          <p className="text-sm text-primary font-medium">Grand Total (After Discount)</p>
                          <p className="text-2xl font-bold font-mono text-primary">₹{calculateDiscountedTotal().toLocaleString()}</p>
                        </div>
                      </div>
                    </div>
                  )}

                  {!discountApplied && (
                    <div className="p-4 bg-primary/10 rounded-lg border border-primary/30">
                      <div className="flex items-center justify-between">
                        <p className="text-sm text-primary font-medium">Grand Total</p>
                        <p className="text-2xl font-bold font-mono text-primary">₹{pricingTotals.grand_total?.toLocaleString()}</p>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </TabsContent>

            <TabsContent value="preview" className="space-y-4">
              <div className="bg-card rounded-xl border border-border p-8">
                <div className="flex items-center justify-between mb-4">
                  <h3 className="font-semibold">Editable Proposal Preview</h3>
                  <Button 
                    onClick={handleSaveBidChanges}
                    disabled={savingBid}
                    className="gap-2"
                  >
                    {savingBid ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="h-4 w-4" />
                        Save Changes
                      </>
                    )}
                  </Button>
                </div>
                <Textarea
                  value={editableBidContent}
                  onChange={(e) => setEditableBidContent(e.target.value)}
                  className="min-h-[600px] font-mono text-sm leading-relaxed"
                  placeholder="Edit your proposal content here..."
                />
                <p className="text-xs text-muted-foreground mt-2">
                  Make any necessary edits to your proposal. Click 'Save Changes' to update. PDF/Word downloads will use the latest saved version.
                </p>
              </div>
            </TabsContent>
          </Tabs>
        </motion.div>

        {/* Version History */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="bg-card rounded-xl border border-border p-4"
        >
          <h3 className="font-semibold mb-4">Version History</h3>
          <div className="space-y-2">
            {versions.map((v) => (
              <div
                key={v.id}
                className="flex items-center justify-between p-3 bg-muted/30 rounded-lg"
              >
                <div className="flex items-center gap-4">
                  <Badge variant="outline">{v.version}</Badge>
                  <span className="text-sm">{v.date}</span>
                  <span className="text-sm text-muted-foreground">by {v.author}</span>
                </div>
                <Badge
                  className={
                    v.status === "Review"
                      ? "bg-info/10 text-info border border-info/30"
                      : "bg-muted text-muted-foreground"
                  }
                >
                  {v.status}
                </Badge>
              </div>
            ))}
          </div>
        </motion.div>

        {/* Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="flex items-center justify-between pt-4"
        >
          <Button variant="outline">
            <Save className="h-4 w-4 mr-2" />
            Save Draft
          </Button>
          <div className="flex gap-3">
            <Button variant="outline" onClick={handleExportPDF}>
              <Download className="h-4 w-4 mr-2" />
              Export PDF
            </Button>
            <Button variant="outline" onClick={handleExportWord}>
              <FileText className="h-4 w-4 mr-2" />
              Export Word
            </Button>
            <Button onClick={handleMarkReadyToSubmit}>
              <Send className="h-4 w-4 mr-2" />
              Mark Ready to Submit
            </Button>
          </div>
        </motion.div>

        {/* Gmail-style Email Modal */}
        <Dialog open={showGmailModal} onOpenChange={setShowGmailModal}>
          <DialogContent className="max-w-3xl max-h-[90vh] overflow-hidden flex flex-col p-0">
            {/* Gmail Header */}
            <div className="bg-[#404040] text-white px-4 py-3 flex items-center justify-between">
              <DialogTitle className="text-base font-medium">New Message</DialogTitle>
              <Button
                variant="ghost"
                size="sm"
                className="h-6 w-6 p-0 hover:bg-white/20 text-white"
                onClick={() => setShowGmailModal(false)}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Gmail Body */}
            <div className="flex-1 overflow-y-auto bg-background">
              <div className="p-4 space-y-3">
                {/* To Field */}
                <div className="flex items-center gap-2 border-b pb-2">
                  <span className="text-sm text-muted-foreground w-16">To:</span>
                  <Input
                    type="email"
                    value={emailData.to}
                    onChange={(e) => setEmailData(prev => ({ ...prev, to: e.target.value }))}
                    className="border-0 focus-visible:ring-0 px-0"
                    placeholder="recipient@example.com"
                  />
                </div>

                {/* Subject Field */}
                <div className="flex items-center gap-2 border-b pb-2">
                  <span className="text-sm text-muted-foreground w-16">Subject:</span>
                  <Input
                    value={emailData.subject}
                    onChange={(e) => setEmailData(prev => ({ ...prev, subject: e.target.value }))}
                    className="border-0 focus-visible:ring-0 px-0"
                    placeholder="Email subject"
                  />
                </div>

                {/* Rich Text Editor */}
                <div className="min-h-[300px]">
                  <ReactQuill
                    theme="snow"
                    value={emailData.body}
                    onChange={(value) => setEmailData(prev => ({ ...prev, body: value }))}
                    modules={{
                      toolbar: [
                        ['bold', 'italic', 'underline'],
                        [{ 'color': [] }],
                        [{ 'list': 'ordered'}, { 'list': 'bullet' }],
                        ['clean']
                      ],
                    }}
                    className="bg-white"
                  />
                </div>

                {/* Attachments */}
                {attachments.length > 0 && (
                  <div className="space-y-2">
                    <p className="text-sm font-medium">Attachments:</p>
                    <div className="space-y-1">
                      {attachments.map((file, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between bg-muted px-3 py-2 rounded-md"
                        >
                          <div className="flex items-center gap-2">
                            <Paperclip className="h-4 w-4 text-muted-foreground" />
                            <span className="text-sm">{file.name}</span>
                            <span className="text-xs text-muted-foreground">
                              ({(file.size / 1024).toFixed(1)} KB)
                            </span>
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => handleRemoveAttachment(index)}
                            className="h-6 w-6 p-0"
                          >
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Gmail Footer */}
            <div className="border-t px-4 py-3 flex items-center justify-between bg-background">
              <div className="flex gap-2">
                <Button
                  onClick={handleSendEmail}
                  disabled={sendingEmail}
                  className="bg-blue-600 hover:bg-blue-700 text-white"
                >
                  {sendingEmail ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Sending...
                    </>
                  ) : (
                    <>
                      <Send className="h-4 w-4 mr-2" />
                      Send
                    </>
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Paperclip className="h-4 w-4 mr-2" />
                  Attach Files
                </Button>
                <input
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".pdf,.doc,.docx,.ppt,.pptx"
                  onChange={handleFileSelect}
                  className="hidden"
                />
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </MainLayout>
  );
}
