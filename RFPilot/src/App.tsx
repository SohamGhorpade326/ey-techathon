import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import { ProtectedRoute } from "@/components/ProtectedRoute";
import Index from "./pages/Index";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import Discovery from "./pages/Discovery";
import RFPDetails from "./pages/RFPDetails";
import Analysis from "./pages/Analysis";
import Pricing from "./pages/Pricing";
import Proposal from "./pages/Proposal";
import Notifications from "./pages/Notifications";
import Repository from "./pages/Repository";
import Settings from "./pages/Settings";
import NotFound from "./pages/NotFound";
import Masteragent from "./pages/Masteragent";
import SkuGapAnalytics from "./pages/SkuGapAnalytics";
const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <Toaster />
      <Sonner />
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Index />} />
          <Route path="/login" element={<Login />} />
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <Dashboard />
              </ProtectedRoute>
            }
          />
          <Route
            path="/discovery"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "sales", "technical"]}>
                <Discovery />
              </ProtectedRoute>
            }
          />
          <Route
            path="/rfp/:rfpId*"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "sales", "technical"]}>
                <RFPDetails />
              </ProtectedRoute>
            }
          />
          <Route
            path="/analysis"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "technical", "product"]}>
                <Analysis />
              </ProtectedRoute>
            }
          />
          <Route
            path="/sku-gap-analytics"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "product", "technical", "pricing"]}>
                <SkuGapAnalytics />
              </ProtectedRoute>
            }
          />
          <Route
            path="/pricing"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "pricing"]}>
                <Pricing />
              </ProtectedRoute>
            }
          />
          <Route
            path="/proposal"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "sales", "pricing"]}>
                <Proposal />
              </ProtectedRoute>
            }
          />
          <Route
            path="/notifications"
            element={
              <ProtectedRoute>
                <Notifications />
              </ProtectedRoute>
            }
          />
          <Route
            path="/repository"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "product"]}>
                <Repository />
              </ProtectedRoute>
            }
          />
          <Route
            path="/settings"
            element={
              <ProtectedRoute allowedRoles={["super_admin"]}>
                <Settings />
              </ProtectedRoute>
            }
          />
          {/* ADD ALL CUSTOM ROUTES ABOVE THE CATCH-ALL "*" ROUTE */}
          <Route
            path="/master-agent"
            element={
              <ProtectedRoute allowedRoles={["super_admin", "sales", "technical", "product"]}>
                <Masteragent />
              </ProtectedRoute>
            }
          />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </BrowserRouter>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
