import Link from "next/link";
import Image from "next/image";

export default function Home() {
  return (
    <div className="flex flex-col bg-background min-h-screen">
      {/* Hero Section */}
      <section className="mx-auto max-w-4xl px-6 pt-16 pb-16 text-center sm:pt-24 sm:pb-20">
        <div className="mx-auto mb-6 flex justify-center">
          <Image
            src="/logo2.png"
            alt="Lumora Logo"
            width={280}
            height={90}
            className="h-16 sm:h-20 w-auto object-contain hover:scale-105 transition-transform duration-300"
            priority
          />
        </div>

        <div className="mx-auto mb-6 inline-flex items-center gap-1.5 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-xs font-semibold text-primary-deep tracking-wider uppercase">
          <span className="h-1.5 w-1.5 rounded-full bg-primary-deep animate-pulse"></span>
          Research Preview
        </div>
        
        <h1 className="text-4xl font-extrabold tracking-tight text-foreground sm:text-5xl md:text-6xl font-sans leading-[1.15]">
          LUMORA <br className="hidden sm:inline" />
          <span className="bg-gradient-to-r from-primary-deep to-primary bg-clip-text text-transparent">
            An Ai Radiology Partner
          </span>
        </h1>
        
        <p className="mx-auto mt-6 max-w-xl text-base sm:text-lg text-muted-foreground leading-relaxed">
          Upload a chest CT, X-ray, or medical report. Lumora flags signs of lung cancer, tuberculosis, and other conditions — with confidence scores you can trust.
        </p>

        <div className="mt-10 flex flex-col items-center justify-center gap-3 sm:flex-row">
          <Link
            href="/analyze"
            className="w-full sm:w-auto inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground shadow-md hover:bg-primary/95 hover:shadow-lg transition-all duration-200 cursor-pointer active:scale-[0.98]"
          >
            Start an analysis
          </Link>
          <Link
            href="#how-it-works"
            className="w-full sm:w-auto inline-flex items-center justify-center rounded-full border border-border bg-background px-8 py-3 text-sm font-semibold text-foreground hover:bg-muted/50 transition-all duration-200 cursor-pointer"
          >
            How it works
          </Link>
        </div>

        <p className="mt-8 text-xs text-muted-foreground font-medium">
          Not a substitute for professional medical diagnosis.
        </p>
      </section>

      {/* Modality Summary Grid */}
      <section className="mx-auto max-w-5xl px-6 py-12 w-full">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
          {/* CT Scan Card */}
          <div className="group rounded-2xl border border-border/60 bg-card p-6 shadow-sm hover:shadow-md hover:border-border transition-all duration-200">
            <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary-deep">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5A3.375 3.375 0 0010.125 2.25H8.25m0 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
              </svg>
            </div>
            <h3 className="text-base font-semibold text-foreground">CT Scan</h3>
            <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
              Upload DICOM, NIfTI, or compressed CT studies for volumetric analysis and reporting.
            </p>
          </div>

          {/* X-Ray Card */}
          <div className="group rounded-2xl border border-border/60 bg-card p-6 shadow-sm hover:shadow-md hover:border-border transition-all duration-200">
            <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary-deep">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            </div>
            <h3 className="text-base font-semibold text-foreground">Chest X-Ray</h3>
            <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
              Drop a single PA or lateral view radiograph and get anatomical findings in seconds.
            </p>
          </div>

          {/* Medical Report Card */}
          <div className="group rounded-2xl border border-border/60 bg-card p-6 shadow-sm hover:shadow-md hover:border-border transition-all duration-200">
            <div className="mb-4 inline-flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary-deep">
              <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-base font-semibold text-foreground">Medical Report</h3>
            <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
              Parse existing radiology or chest test reports for structured clinical insights.
            </p>
          </div>
        </div>
      </section>

      {/* "Three steps. No clutter." Workflow Section */}
      <section id="how-it-works" className="border-t border-border/60 bg-surface py-20 w-full scroll-mt-16">
        <div className="mx-auto max-w-5xl px-6">
          <div className="flex flex-col items-center justify-between gap-6 md:flex-row md:items-end mb-12">
            <div className="max-w-md text-center md:text-left">
              <div className="text-xs font-bold text-primary-deep tracking-wider uppercase mb-2">Workflow</div>
              <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
                Three steps. No clutter.
              </h2>
            </div>
            <Link
              href="/analyze"
              className="inline-flex items-center gap-1 text-sm font-semibold text-primary-deep hover:text-primary transition-colors cursor-pointer"
            >
              Read the pipeline &rarr;
            </Link>
          </div>

          <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
            {/* Step 1 */}
            <div className="relative flex flex-col items-center text-center md:items-start md:text-left">
              <span className="text-4xl font-black text-primary/25 font-mono mb-3">01</span>
              <h3 className="text-lg font-bold text-foreground">Upload</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-sm">
                Pick CT, X-ray, or a report. Drag, drop, done.
              </p>
            </div>

            {/* Step 2 */}
            <div className="relative flex flex-col items-center text-center md:items-start md:text-left">
              <span className="text-4xl font-black text-primary/25 font-mono mb-3">02</span>
              <h3 className="text-lg font-bold text-foreground">Analyze</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-sm">
                Our models preprocess and read the study in seconds.
              </p>
            </div>

            {/* Step 3 */}
            <div className="relative flex flex-col items-center text-center md:items-start md:text-left">
              <span className="text-4xl font-black text-primary/25 font-mono mb-3">03</span>
              <h3 className="text-lg font-bold text-foreground">Report</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-sm">
                Get a clear prediction, confidence scores, and key findings.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Built for clinical decision support Section */}
      <section id="about" className="py-20 w-full scroll-mt-16 bg-background">
        <div className="mx-auto max-w-5xl px-6">
          <div className="text-center mb-16">
            <div className="text-xs font-bold text-primary-deep tracking-wider uppercase mb-2">System Architecture</div>
            <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
              Built for clinical decision support
            </h2>
          </div>

          <div className="grid grid-cols-1 gap-8 sm:grid-cols-3">
            {/* Privacy Card */}
            <div className="flex flex-col items-center text-center p-6 bg-surface/50 border border-border/50 rounded-2xl">
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-primary/5 text-primary-deep">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <h3 className="text-base font-bold text-foreground">Privacy-first</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-xs">
                Studies are processed in-session. No images leave your browser without explicit clinical consent.
              </p>
            </div>

            {/* Verification Card */}
            <div className="flex flex-col items-center text-center p-6 bg-surface/50 border border-border/50 rounded-2xl">
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-primary/5 text-primary-deep">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-base font-bold text-foreground">Radiologist-reviewed</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-xs">
                Models are validated against ground truth board-certified radiologist reads from standard datasets.
              </p>
            </div>

            {/* Clarity Card */}
            <div className="flex flex-col items-center text-center p-6 bg-surface/50 border border-border/50 rounded-2xl">
              <div className="mb-4 inline-flex h-12 w-12 items-center justify-center rounded-full bg-primary/5 text-primary-deep">
                <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h3 className="text-base font-bold text-foreground">Designed for clarity</h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed max-w-xs">
                Provides plain-language explanations of findings alongside standard diagnostic nomenclature.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Ready when you are CTA Section */}
      <section className="mx-auto max-w-5xl px-6 py-12 w-full mb-16">
        <div className="relative overflow-hidden rounded-3xl bg-surface border border-border/60 px-6 py-16 text-center sm:px-16 sm:py-20 shadow-sm">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0">
            <div className="h-64 w-64 rounded-full bg-primary/5 blur-3xl"></div>
          </div>
          <div className="relative z-10 mx-auto max-w-md">
            <h2 className="text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
              Ready when you are.
            </h2>
            <p className="mt-4 text-sm sm:text-base text-muted-foreground">
              Try Lumora with a sample study or your own chest imaging files.
            </p>
            <div className="mt-8 flex justify-center">
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground shadow-md hover:bg-primary/95 transition-all duration-200 cursor-pointer"
              >
                Open the analyzer
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
