import { motion } from "framer-motion";
import { Bell, Search, User, ChevronRight } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface TopBarProps {
  title: string;
  breadcrumbs?: { name: string; href?: string }[];
}

export function TopBar({ title, breadcrumbs = [] }: TopBarProps) {
  return (
    <motion.header
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className="sticky top-0 z-30 flex h-16 items-center justify-between border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6"
    >
      {/* Left: Breadcrumbs & Title */}
      <div className="flex items-center gap-4">
        {breadcrumbs.length > 0 && (
          <nav className="flex items-center gap-1 text-sm text-muted-foreground">
            {breadcrumbs.map((crumb, index) => (
              <span key={crumb.name} className="flex items-center gap-1">
                {index > 0 && <ChevronRight className="h-4 w-4" />}
                {crumb.href ? (
                  <a href={crumb.href} className="hover:text-foreground transition-colors">
                    {crumb.name}
                  </a>
                ) : (
                  <span className="text-foreground font-medium">{crumb.name}</span>
                )}
              </span>
            ))}
          </nav>
        )}
        {!breadcrumbs.length && (
          <h1 className="text-xl font-semibold text-foreground">{title}</h1>
        )}
      </div>

      {/* Right: Search + Actions */}
      <div className="flex items-center gap-4">
        <div className="relative hidden md:block">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search RFPs, SKUs..."
            className="w-64 pl-9 bg-secondary/50 border-0 focus-visible:ring-1"
          />
        </div>

        <Button variant="ghost" size="icon" className="relative">
          <Bell className="h-5 w-5" />
          <span className="absolute -top-0.5 -right-0.5 flex h-4 w-4 items-center justify-center rounded-full bg-destructive text-[10px] font-bold text-destructive-foreground">
            3
          </span>
        </Button>

        <Button variant="ghost" size="icon" className="rounded-full">
          <User className="h-5 w-5" />
        </Button>
      </div>
    </motion.header>
  );
}
