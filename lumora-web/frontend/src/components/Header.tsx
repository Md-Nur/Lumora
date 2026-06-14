"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";

type NavPage = "/" | "/workspace" | "/models" | "/comparison";

interface HeaderProps {
  activePage: NavPage;
  /** Optional: pass backend status for workspace page */
  backendStatus?: "checking" | "online" | "offline";
}

const NAV_LINKS: { href: NavPage; label: string }[] = [
  { href: "/", label: "Portal Home" },
  { href: "/workspace", label: "Clinical Workspace" },
  { href: "/models", label: "Model Specs" },
  { href: "/comparison", label: "Benchmark" },
];

export default function Header({ activePage, backendStatus }: HeaderProps) {
  const [menuOpen, setMenuOpen] = useState(false);

  // Close menu on route change / resize
  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 640) setMenuOpen(false);
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  // Prevent body scroll when menu is open on mobile
  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => { document.body.style.overflow = ""; };
  }, [menuOpen]);

  return (
    <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md sticky top-0 z-50 shadow-[0_1px_4px_rgba(0,0,0,0.05)]">
      <div className="max-w-7xl mx-auto px-3 sm:px-6 py-2.5 flex items-center justify-between gap-2">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center shrink-0 h-10"
          onClick={() => setMenuOpen(false)}
        >
          <img
            src="/logo.png"
            alt="Lumora Logo"
            className="h-9 sm:h-10 w-auto object-contain hover:brightness-105 transition-all"
          />
        </Link>

        {/* Desktop Nav */}
        <nav className="hidden sm:flex items-center gap-1 bg-slate-100 rounded-lg p-0.5 border border-slate-200 flex-1 max-w-[480px] ml-2">
          {NAV_LINKS.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className={`flex-1 text-center px-2 py-1.5 rounded-md text-[11px] font-bold transition-all whitespace-nowrap ${
                activePage === href
                  ? "bg-white text-blue-600 shadow-xs border border-blue-100"
                  : "text-slate-500 hover:text-slate-700"
              }`}
            >
              {label}
            </Link>
          ))}
        </nav>

        {/* Right side: backend status (desktop) + hamburger */}
        <div className="flex items-center gap-2 ml-auto shrink-0">
          {/* Backend status — only shown on desktop in workspace */}
          {backendStatus && (
            <div className="hidden sm:flex items-center gap-2 px-2.5 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs font-medium">
              {backendStatus === "checking" && (
                <span className="flex items-center gap-1.5 text-slate-500">
                  <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse" />
                  <span className="hidden md:inline">Connecting...</span>
                </span>
              )}
              {backendStatus === "online" && (
                <span className="flex items-center gap-1.5 text-emerald-700">
                  <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping" />
                  <span className="hidden md:inline">System Ready</span>
                </span>
              )}
              {backendStatus === "offline" && (
                <span className="flex items-center gap-1.5 text-amber-700">
                  <span className="h-2 w-2 rounded-full bg-amber-500" />
                  <span className="hidden md:inline">Demo Mode</span>
                </span>
              )}
            </div>
          )}

          {/* Hamburger button — visible only on small screens */}
          <button
            type="button"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((v) => !v)}
            className="sm:hidden p-2 rounded-lg border border-slate-200 bg-white text-slate-600 hover:bg-slate-50 transition-all active:scale-95"
          >
            {menuOpen ? (
              // X icon
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              // Hamburger icon
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Mobile slide-down menu */}
      {menuOpen && (
        <div className="sm:hidden border-t border-slate-200 bg-white/98 backdrop-blur-md shadow-lg">
          <nav className="flex flex-col divide-y divide-slate-100">
            {NAV_LINKS.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                onClick={() => setMenuOpen(false)}
                className={`px-5 py-3.5 text-sm font-bold transition-colors ${
                  activePage === href
                    ? "text-blue-600 bg-blue-50/70"
                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                }`}
              >
                <span className="flex items-center gap-2">
                  {activePage === href && (
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-500" />
                  )}
                  {label}
                </span>
              </Link>
            ))}
          </nav>

          {/* Backend status in mobile menu */}
          {backendStatus && (
            <div className="px-5 py-3 border-t border-slate-100 flex items-center gap-2 text-xs font-medium text-slate-500">
              {backendStatus === "checking" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse shrink-0" />
                  Connecting to inference engine...
                </>
              )}
              {backendStatus === "online" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-emerald-500 shrink-0" />
                  <span className="text-emerald-700">Inference System: Ready</span>
                </>
              )}
              {backendStatus === "offline" && (
                <>
                  <span className="h-2 w-2 rounded-full bg-amber-500 shrink-0" />
                  <span className="text-amber-700">Demo Simulator Active</span>
                </>
              )}
            </div>
          )}
        </div>
      )}
    </header>
  );
}
