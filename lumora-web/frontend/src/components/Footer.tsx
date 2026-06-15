import Link from "next/link";

const FOOTER_LINKS = [
  { href: "/", label: "Home" },
  { href: "/workspace", label: "Analyze" },
  { href: "/models", label: "Models" },
  { href: "/comparison", label: "Comparison" },
] as const;

export default function Footer() {
  return (
    <footer className="border-t border-slate-200 bg-white py-8">
      <div className="mx-auto max-w-6xl px-3 sm:px-6">
        <div className="mb-6 flex flex-col items-center justify-between gap-6 sm:flex-row">
          <div className="flex flex-wrap items-center justify-center gap-x-4 gap-y-2 text-sm font-medium text-slate-600">
            {FOOTER_LINKS.map(({ href, label }, index) => (
              <span key={href} className="flex items-center gap-4">
                <Link href={href} className="text-blue-600 transition-colors hover:text-blue-700">
                  {label}
                </Link>
                {index < FOOTER_LINKS.length - 1 && <span className="text-slate-300">•</span>}
              </span>
            ))}
          </div>
        </div>

        <div className="border-t border-slate-100 pt-4 text-center">
          <p className="text-[10px] font-medium tracking-wide text-slate-400">
            LUMORA CLINICAL DECISION-SUPPORT SYSTEM — DESIGNED FOR LICENSED HEALTHCARE PROVIDERS
          </p>
        </div>
      </div>
    </footer>
  );
}