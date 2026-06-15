"use client";

import React from "react";
import Link from "next/link";

export default function About() {
  return (
    <div className="flex flex-col bg-background min-h-screen">
      <main className="mx-auto max-w-4xl px-6 py-16 w-full">
        
        {/* Hero Section - Focused on Lumora */}
        <div className="text-center md:text-left mb-16">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-[10px] font-bold text-primary-deep tracking-wider uppercase mb-3">
            System Profile
          </div>
          <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-foreground font-sans leading-tight">
            An Intelligent AI Partner <br />
            for Chest Radiology
          </h1>
          <p className="mt-4 max-w-2xl text-muted-foreground leading-relaxed text-sm sm:text-base">
            Lumora is an advanced multimodal diagnostic assistant engineered to generate clinical radiology reports, verify scan usability, and explain complex findings in patient-friendly language.
          </p>
        </div>

        {/* Core Capabilities Section */}
        <section className="border-t border-border/60 py-12">
          <div className="mb-8">
            <h2 className="text-xl font-bold tracking-tight text-slate-800">
              Core Capabilities & Architecture
            </h2>
            <p className="text-xs text-muted-foreground mt-1">
              Three specialized AI layers working together in a unified pipeline
            </p>
          </div>

          <div className="grid gap-6 sm:grid-cols-1 md:grid-cols-3">
            
            {/* VLM Card */}
            <div className="rounded-2xl border border-border bg-white p-6 shadow-2xs hover:shadow-xs hover:border-slate-350 transition-all duration-200">
              <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary-deep flex items-center justify-center mb-4">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              </div>
              <h3 className="text-sm font-bold text-slate-800">Draft Report Generation</h3>
              <p className="mt-2 text-xs text-slate-500 leading-relaxed">
                Uses custom Vision-Language Models (VLM) pre-trained on clinical datasets (MIMIC-CXR and CT-RATE) to draft structured radiology findings from chest X-rays and 2D CT scan slices.
              </p>
            </div>

            {/* YOLO Card */}
            <div className="rounded-2xl border border-border bg-white p-6 shadow-2xs hover:shadow-xs hover:border-slate-350 transition-all duration-200">
              <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary-deep flex items-center justify-center mb-4">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
                </svg>
              </div>
              <h3 className="text-sm font-bold text-slate-800">Physics Guardrails</h3>
              <p className="mt-2 text-xs text-slate-500 leading-relaxed">
                Monitors pixel distributions, gray scale contrast, and saturation thresholds in real time using a YOLO-based classifier to verify that inputs are genuine medical scan modalities.
              </p>
            </div>

            {/* Translation Card */}
            <div className="rounded-2xl border border-border bg-white p-6 shadow-2xs hover:shadow-xs hover:border-slate-350 transition-all duration-200">
              <div className="h-10 w-10 rounded-xl bg-primary/10 text-primary-deep flex items-center justify-center mb-4">
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z" />
                </svg>
              </div>
              <h3 className="text-sm font-bold text-slate-800">Layperson Translation</h3>
              <p className="mt-2 text-xs text-slate-500 leading-relaxed">
                Translates complex diagnostic terminology into clear, reassuring language for patients using a fine-tuned T5 adapter network, preventing patient anxiety and clinical ambiguity.
              </p>
            </div>

          </div>
        </section>

        {/* Future Vision - PACS Integration */}
        <section className="border-t border-border/60 py-12">
          <div className="grid gap-8 sm:grid-cols-1 md:grid-cols-12 items-center">
            
            <div className="md:col-span-7">
              <span className="text-[10px] font-bold text-primary-deep uppercase tracking-wider font-mono">
                System Integration
              </span>
              <h2 className="text-xl font-bold tracking-tight text-slate-800 mt-1">
                Clinical Workflow Vision
              </h2>
              <p className="mt-3 text-xs sm:text-sm text-slate-500 leading-relaxed">
                Lumora is envisioned to operate beyond the bounds of a web preview. Our future plan is to integrate this assistant directly into actual physical X-ray machines and CT scanner consoles.
              </p>
              <p className="mt-2.5 text-xs sm:text-sm text-slate-500 leading-relaxed">
                By connecting the pipeline with PACS (Picture Archiving and Communication Systems) and scanner hardware suites, we aim to provide real-time diagnostic decision support directly at the point of care.
              </p>
            </div>

            <div className="md:col-span-5 bg-surface rounded-2xl border border-border p-6">
              <h3 className="text-xs font-bold font-mono text-slate-700 uppercase mb-3">PACS Node Pipeline</h3>
              <div className="flex flex-col gap-2.5 text-[10px] font-mono leading-normal">
                <div className="flex items-center gap-2 text-slate-650 bg-white border border-border/70 p-2 rounded-lg">
                  <span className="h-2 w-2 rounded-full bg-primary-deep" />
                  1. Scan Capture & DICOM Push
                </div>
                <div className="flex items-center gap-2 text-slate-650 bg-white border border-border/70 p-2 rounded-lg">
                  <span className="h-2 w-2 rounded-full bg-primary-deep" />
                  2. Lumora Ingestion & Guardrail Check
                </div>
                <div className="flex items-center gap-2 text-slate-650 bg-white border border-border/70 p-2 rounded-lg">
                  <span className="h-2 w-2 rounded-full bg-primary-deep" />
                  3. Real-time Report Draft & Overlay
                </div>
              </div>
            </div>

          </div>
        </section>

        {/* Development & Academic Credits - Consolidating student details */}
        <section className="border-t border-border/60 py-12">
          <div className="rounded-2xl border border-border bg-white p-6 shadow-2xs">
            <span className="text-[10px] font-bold text-primary-deep uppercase tracking-wider font-mono">
              Academic Preview
            </span>
            <h2 className="text-base font-bold text-slate-800 mt-1">
              Developed by University of Rajshahi
            </h2>
            <p className="mt-3 text-xs sm:text-sm text-slate-500 leading-relaxed">
              Lumora was created as a research project by three students of the <strong>Department of Electrical and Electronic Engineering (EEE)</strong> at the <strong>University of Rajshahi</strong>:
            </p>
            
            {/* Student list */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4 border-t border-border/40 pt-4">
              <div>
                <p className="text-xs font-bold text-slate-800">1. Md. Nur E Alam Siddiquee</p>
                <p className="text-[10px] text-slate-500 font-medium">EEE Department, RU</p>
              </div>
              <div>
                <p className="text-xs font-bold text-slate-800">2. Mozammel Haque</p>
                <p className="text-[10px] text-slate-500 font-medium">EEE Department, RU</p>
              </div>
              <div>
                <p className="text-xs font-bold text-slate-800">3. Pranto Protim Dutta</p>
                <p className="text-[10px] text-slate-500 font-medium">EEE Department, RU</p>
              </div>
            </div>

            <p className="mt-4 text-[11px] text-slate-400 leading-normal border-t border-border/40 pt-3">
              The project demonstrates the synergy between traditional signal processing methodologies and modern vision-language networks applied to medical diagnostic support systems.
            </p>
          </div>
        </section>

        {/* CTA section */}
        <section className="mt-4 relative overflow-hidden rounded-3xl bg-surface border border-border px-6 py-12 text-center sm:px-12 sm:py-16 shadow-2xs">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0">
            <div className="h-48 w-48 rounded-full bg-primary/5 blur-3xl"></div>
          </div>
          <div className="relative z-10 mx-auto max-w-md">
            <h2 className="text-xl font-bold tracking-tight text-foreground sm:text-2xl">
              Explore Lumora
            </h2>
            <p className="mt-2.5 text-xs text-muted-foreground leading-relaxed">
              Try analyzing a 2D scan slice using one of our preloaded sample files in the workspace.
            </p>
            <div className="mt-6 flex justify-center gap-3">
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center rounded-full bg-primary px-7 py-2.5 text-xs font-bold text-primary-foreground shadow-sm hover:bg-primary/95 transition-all duration-200 cursor-pointer"
              >
                Go to Workspace
              </Link>
              <Link
                href="/how-it-works"
                className="inline-flex items-center justify-center rounded-full border border-border bg-background px-6 py-2.5 text-xs font-bold text-foreground hover:bg-muted/50 transition-all duration-200 cursor-pointer"
              >
                Learn More
              </Link>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}
