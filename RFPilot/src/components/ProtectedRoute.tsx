import { Navigate } from "react-router-dom";
import { ReactNode } from "react";
import { Button } from "@/components/ui/button";

interface ProtectedRouteProps {
  children: ReactNode;
  allowedRoles?: string[];
}

export function ProtectedRoute({ children, allowedRoles }: ProtectedRouteProps) {
  const token = localStorage.getItem("token");
  const userStr = localStorage.getItem("user");

  // Check if user is logged in
  if (!token || !userStr) {
    return <Navigate to="/login" replace />;
  }

  // If no roles specified, allow all authenticated users
  if (!allowedRoles || allowedRoles.length === 0) {
    return <>{children}</>;
  }

  // Check if user has required role
  const user = JSON.parse(userStr);
  
  // Debug logging
  console.log("ProtectedRoute - User role:", user.role);
  console.log("ProtectedRoute - Allowed roles:", allowedRoles);
  console.log("ProtectedRoute - Role type:", typeof user.role);
  console.log("ProtectedRoute - Access granted:", allowedRoles.includes(user.role));
  
  if (!allowedRoles.includes(user.role)) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-background">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-destructive">Access Denied</h1>
          <p className="text-muted-foreground">
            You don't have permission to access this page.
          </p>
          <p className="text-sm text-muted-foreground">
            Your role: <span className="font-semibold">"{user.role}"</span> (type: {typeof user.role})
          </p>
          <p className="text-sm text-muted-foreground">
            Required roles: <span className="font-semibold">{allowedRoles.join(", ")}</span>
          </p>
          <Button onClick={() => window.location.href = "/dashboard"} className="mt-4">
            Go to Dashboard
          </Button>
        </div>
      </div>
    );
  }

  return <>{children}</>;
}
