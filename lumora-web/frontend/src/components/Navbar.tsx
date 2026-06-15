"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import Image from "next/image";

export default function Navbar() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  useEffect(() => {
    const onResize = () => {
      if (window.innerWidth >= 768) setMenuOpen(false);
    };

    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  useEffect(() => {
    document.body.style.overflow = menuOpen ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [menuOpen]);

  const isActive = (href: string) => {
    if (href === "/") return pathname === "/";
    return pathname === href || pathname.startsWith(href + "/");
  };

  return (
    <header className="sticky top-0 z-40 w-full border-b border-border/60 bg-background/80 backdrop-blur">
      <div className="mx-auto flex h-20 max-w-6xl items-center justify-between px-6">
        {/* Logo (enlarged, text removed) */}
        <Link href="/" className="flex items-center transition-opacity hover:opacity-90">
          <Image
            src="/logo.png"
            alt="Lumora Logo"
            className="h-40 w-40 object-contain"
            height={200}
            width={200}
          />
        </Link>

        {/* Desktop Navigation Links */}
        <nav className="hidden items-center gap-8 text-sm md:flex">
          <Link
            href="/analyze"
            className={`transition-colors font-medium ${
              isActive("/analyze")
                ? "text-primary-deep font-semibold"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            Analyze
          </Link>
          <Link
            href="/how-it-works"
            className={`transition-colors font-medium ${
              isActive("/how-it-works")
                ? "text-primary-deep font-semibold"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            How it works
          </Link>
          <Link
            href="/about"
            className={`transition-colors font-medium ${
              isActive("/about")
                ? "text-primary-deep font-semibold"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            About us
          </Link>
        </nav>

        {/* Action Buttons */}
        <div className="flex items-center gap-3">
          <Link
            href="/login"
            className="hidden text-sm font-medium text-muted-foreground transition-colors hover:text-foreground sm:inline"
          >
            Sign in
          </Link>
          <Link
            href="/analyze"
            className="inline-flex items-center justify-center gap-2 whitespace-nowrap text-xs font-semibold cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50 disabled:cursor-not-allowed bg-primary text-primary-foreground shadow hover:bg-primary/90 h-8 rounded-full px-4"
          >
            Try now
          </Link>

          {/* Hamburger Menu Toggle (Mobile) */}
          <button
            type="button"
            aria-label={menuOpen ? "Close menu" : "Open menu"}
            aria-expanded={menuOpen}
            onClick={() => setMenuOpen((value) => !value)}
            className="rounded-full border border-border p-2 text-muted-foreground transition-colors hover:bg-muted active:scale-95 md:hidden cursor-pointer"
          >
            {menuOpen ? (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            ) : (
              <svg className="h-4 w-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            )}
          </button>
        </div>
      </div>

      {/* Mobile Slide-down Menu */}
      {menuOpen && (
        <div className="border-t border-border/60 bg-background/98 shadow-lg md:hidden animate-in fade-in slide-in-from-top-4 duration-200">
          <nav className="flex flex-col p-4 gap-3">
            <Link
              href="/analyze"
              onClick={() => setMenuOpen(false)}
              className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                isActive("/analyze")
                  ? "bg-primary/10 text-primary-deep font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
              }`}
            >
              Analyze
            </Link>
            <Link
              href="/how-it-works"
              onClick={() => setMenuOpen(false)}
              className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                isActive("/how-it-works")
                  ? "bg-primary/10 text-primary-deep font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
              }`}
            >
              How it works
            </Link>
            <Link
              href="/about"
              onClick={() => setMenuOpen(false)}
              className={`px-3 py-2 text-sm font-medium rounded-lg transition-colors ${
                isActive("/about")
                  ? "bg-primary/10 text-primary-deep font-semibold"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
              }`}
            >
              About us
            </Link>
            <hr className="border-border/60 my-1" />
            <div className="flex flex-col gap-2 px-3 pt-2">
              <Link
                href="/login"
                onClick={() => setMenuOpen(false)}
                className="text-center py-2 text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted/50 rounded-lg transition-colors"
              >
                Sign in
              </Link>
              <Link
                href="/analyze"
                onClick={() => setMenuOpen(false)}
                className="text-center py-2.5 text-xs font-semibold bg-primary text-primary-foreground hover:bg-primary/90 rounded-full transition-colors"
              >
                Try now
              </Link>
            </div>
          </nav>
        </div>
      )}
    </header>
  );
}