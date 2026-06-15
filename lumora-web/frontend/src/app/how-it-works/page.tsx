"use client";

import React, { useState } from "react";
import Link from "next/link";

const PIPELINE_STEPS = [
  {
    num: "01",
    title: "Secure upload",
    desc: "Studies are sent over an encrypted channel. CT volumes, X-rays, and reports are accepted in standard formats.",
  },
  {
    num: "02",
    title: "Preprocessing",
    desc: "Images are normalized, registered, and cropped to lung fields. Reports are parsed into structured features.",
  },
  {
    num: "03",
    title: "Model inference",
    desc: "Specialized models score each study against lung cancer, tuberculosis, pneumonia, and normal classes.",
  },
  {
    num: "04",
    title: "Human-ready report",
    desc: "Findings are paired with confidence and a differential — designed to be skimmed in under 30 seconds.",
  },
];

const FAQ_ITEMS = [
  {
    q: "Is Lumora a medical diagnosis?",
    a: "No. Lumora is a research and triage assistant. All findings should be reviewed by a licensed radiologist or clinician.",
  },
  {
    q: "What file formats are supported?",
    a: "CT: DICOM (.dcm), NIfTI (.nii), ZIP studies, PNG/JPG slices. X-ray: PNG, JPG. Reports: PDF, DOCX, TXT, scans.",
  },
  {
    q: "How is my data handled?",
    a: "In this preview, files are processed locally in your browser session and discarded when you leave the page.",
  },
  {
    q: "How accurate are the models?",
    a: "Internal validation shows AUROC > 0.92 on benchmark chest X-ray and CT datasets. Real-world performance varies.",
  },
];

export default function HowItWorks() {
  const [openFaq, setOpenFaq] = useState<number | null>(null);

  const toggleFaq = (index: number) => {
    setOpenFaq(openFaq === index ? null : index);
  };

  return (
    <div className="flex flex-col bg-background min-h-screen">
      <main className="mx-auto max-w-5xl px-6 py-20 w-full">
        {/* Header Section */}
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-primary-deep">
            Pipeline
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl text-foreground">
            How Lumora reads a study.
          </h1>
          <p className="mt-4 max-w-xl text-muted-foreground">
            A focused four-step pipeline turns raw uploads into a clean, confidence-aware report.
          </p>
        </div>

        {/* Pipeline Steps Grid */}
        <div className="mt-12 grid gap-px overflow-hidden rounded-3xl border border-border bg-border sm:grid-cols-2">
          {PIPELINE_STEPS.map((step, idx) => (
            <div key={idx} className="bg-card p-8 shadow-2xs hover:bg-surface/30 transition-colors duration-150">
              <span className="text-xs font-semibold tracking-[0.2em] text-primary">
                {step.num}
              </span>
              <h3 className="mt-3 text-xl font-semibold text-foreground">
                {step.title}
              </h3>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                {step.desc}
              </p>
            </div>
          ))}
        </div>

        {/* FAQ Section */}
        <section className="mt-20">
          <h2 className="text-2xl font-semibold tracking-tight text-foreground border-b border-border/40 pb-4">
            FAQ
          </h2>
          
          <div className="mt-4 flex flex-col">
            {FAQ_ITEMS.map((item, idx) => {
              const isOpen = openFaq === idx;
              return (
                <div key={idx} className="border-b border-border/60">
                  <h3>
                    <button
                      type="button"
                      onClick={() => toggleFaq(idx)}
                      aria-expanded={isOpen}
                      className="flex w-full items-center justify-between py-4 text-sm font-semibold text-foreground transition-all hover:underline text-left cursor-pointer outline-none select-none"
                    >
                      <span>{item.q}</span>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        className={`h-4 w-4 shrink-0 text-muted-foreground transition-transform duration-200 ${
                          isOpen ? "rotate-180" : ""
                        }`}
                      >
                        <path d="m6 9 6 6 6-6"></path>
                      </svg>
                    </button>
                  </h3>
                  
                  {/* Dynamic Accordion Slide Content */}
                  <div
                    className={`overflow-hidden text-sm transition-all duration-200 ${
                      isOpen ? "max-h-32 opacity-100 pb-4" : "max-h-0 opacity-0"
                    }`}
                  >
                    <p className="text-muted-foreground leading-relaxed">
                      {item.a}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </section>
      </main>
    </div>
  );
}
