import type { Metadata } from "next";
import type { ReactNode } from "react";
import "./globals.css";

export const metadata: Metadata = {
  title: "Superoptimising Synthesis Stack",
  description:
    "Stochastic superoptimisation, RL/MCTS assembly synthesis, equality saturation with ILP extraction, SMT verification and a Zig GPU runtime.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-slate-950 text-slate-200 antialiased">{children}</body>
    </html>
  );
}
