import Link from "next/link";

const FOOTER_LINKS = [
  { href: "/models", label: "Models" },
  { href: "/comparison", label: "Comparison" },
] as const;

export default function Footer() {
  return (
    <footer className="border-t border-border/60 bg-background py-8">
      <div className="mx-auto max-w-6xl px-6">
        <div className="flex flex-col items-center justify-between gap-4 md:flex-row">
          {/* Footer Navigation Links */}
          <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2 text-sm text-muted-foreground">
            {FOOTER_LINKS.map(({ href, label }) => (
              <Link
                key={href}
                href={href}
                className="transition-colors hover:text-foreground font-medium"
              >
                {label}
              </Link>
            ))}
          </div>
          {/* Footer Copyright and Disclaimer */}
          <p className="text-xs text-muted-foreground md:text-right">
            &copy; 2026 Lumora. Not a medical device.
          </p>
        </div>
      </div>
    </footer>
  );
}