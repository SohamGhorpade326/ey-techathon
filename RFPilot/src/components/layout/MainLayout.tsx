import { ReactNode } from "react";
import { AppSidebar } from "./AppSidebar";
import { TopBar } from "./TopBar";

interface MainLayoutProps {
  children: ReactNode;
  title: string;
  breadcrumbs?: { name: string; href?: string }[];
}

export function MainLayout({ children, title, breadcrumbs }: MainLayoutProps) {
  return (
    <div className="min-h-screen flex w-full bg-background">
      <AppSidebar />
      <div className="flex-1 ml-[280px]">
        <TopBar title={title} breadcrumbs={breadcrumbs} />
        <main className="p-6">{children}</main>
      </div>
    </div>
  );
}
