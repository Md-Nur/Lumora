"use client";

import React from "react";
import Link from "next/link";

const TECH_CARDS = [
  {
    title: "Chest X-Ray VLM",
    desc: "A multimodal encoder-decoder architecture mapping DenseNet-121 visual features to GPT-2 language space, generating comprehensive clinical findings.",
    tag: "Radiology VLM",
    image: "/xray_vlm_metrics.png",
    caption: "Validation loss convergence and BLEU-4 score growth over fine-tuning epochs on the MIMIC-CXR dataset.",
  },
  {
    title: "CT-RATE Volumetric Model",
    desc: "Processes 3D volumetric chest CT scans via axial, coronal, and sagittal projections to detect abnormalities and output detailed anatomical summaries.",
    tag: "Volumetric VLM",
    image: "/ct_vlm_metrics.png",
    caption: "CT-RATE training convergence showing loss optimization and metrics improvement during local phase 2 runs.",
  },
  {
    title: "Layperson Translation",
    desc: "A fine-tuned T5-small adapter that processes medical report text and output descriptions to translate clinical jargon into clear, layperson terms.",
    tag: "T5 Adapter",
    image: "/translation_metrics.png",
    caption: "BLEU and ROUGE translation metric improvements on the medical-to-layman parallel corpus.",
  },
  {
    title: "Disease Classifier",
    desc: "A high-recall Bio_ClinicalBERT classifier trained on clinical report transcripts to detect 26 canonical chest/lung pathologies.",
    tag: "Pathology Detection",
    image: "/disease_detection_metrics.png",
    caption: "ROC curves showing Area Under ROC (AUROC) performance across multiple multi-label disease targets.",
  },
  {
    title: "YOLO Modality Guardrail",
    desc: "A fast, light YOLO-based image classification module that verifies scan integrity, filtering out invalid or non-diagnostic visuals.",
    tag: "Physics Guardrail",
    image: "/yolo_guardrail_metrics.png",
    caption: "Confusion matrix showing near-perfect separation between chest X-rays, CT slices, and blank or natural images.",
  },
];

export default function About() {
  return (
    <div className="flex flex-col bg-background min-h-screen">
      <main className="mx-auto max-w-6xl px-6 py-20 w-full">
        {/* Hero Section */}
        <div className="text-center sm:text-left mb-16">
          <p className="text-xs uppercase tracking-[0.2em] text-primary-deep font-semibold">
            About Lumora
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl text-foreground">
            Clear medical insights for everyone.
          </h1>
          <p className="mt-4 max-w-2xl text-muted-foreground leading-relaxed">
            Lumora is an advanced AI radiology partner designed to generate clinical report summaries, detect pathologies, and translate complex medical jargon into friendly, accessible descriptions.
          </p>
        </div>

        {/* Mission / Context Section */}
        <section className="grid gap-12 sm:grid-cols-1 md:grid-cols-2 border-t border-border/60 py-12">
          <div>
            <h2 className="text-xl font-semibold text-foreground">Our Mission</h2>
            <p className="mt-4 text-sm text-muted-foreground leading-relaxed">
              Medical reports and radiology scans are highly technical documents filled with clinical jargon. This creates a significant gap of understanding for patients trying to review their own health files.
            </p>
            <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
              Lumora bridges this gap. By using state-of-the-art vision-language models alongside dedicated disease classification and translation adapters, Lumora translates visual diagnostic scans and clinical report transcripts into structured, patient-friendly information.
            </p>
          </div>
          <div className="bg-card rounded-2xl border border-border p-6 sm:p-8 flex flex-col justify-center">
            <h3 className="text-base font-semibold text-foreground mb-3">Core Commitments</h3>
            <ul className="space-y-3 text-sm text-muted-foreground">
              <li className="flex items-start gap-2.5">
                <svg className="h-5 w-5 text-primary shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                <span><strong>Explainability:</strong> Giving patients a clear understanding of clinical terms.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <svg className="h-5 w-5 text-primary shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                <span><strong>High Recall:</strong> Sensitive guardrails to make sure potential conditions are flagged.</span>
              </li>
              <li className="flex items-start gap-2.5">
                <svg className="h-5 w-5 text-primary shrink-0 mt-0.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
                <span><strong>Privacy-first:</strong> In-session analysis with zero persistent cloud image storage.</span>
              </li>
            </ul>
          </div>
        </section>

        {/* Model Metrics & Tech Section */}
        <section className="mt-12 border-t border-border/60 pt-16">
          <div className="text-center md:text-left mb-10">
            <h2 className="text-2xl font-semibold tracking-tight text-foreground">
              Core Technologies & Verification Metrics
            </h2>
            <p className="mt-2 text-sm text-muted-foreground max-w-xl">
              We validate our neural networks against standard medical datasets and radiologists' ground truths. Here is a look at the training metrics and pipeline performance charts.
            </p>
          </div>

          {/* Metrics Grid */}
          <div className="grid gap-8 sm:grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
            {TECH_CARDS.map((card, idx) => (
              <div
                key={idx}
                className="group flex flex-col rounded-3xl border border-border bg-card overflow-hidden hover:shadow-lg hover:border-primary/45 transition-all duration-300"
              >
                {/* Header Tag / Title */}
                <div className="p-6 pb-4 flex-1">
                  <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-primary/10 text-primary-deep mb-3">
                    {card.tag}
                  </span>
                  <h3 className="text-lg font-bold text-foreground group-hover:text-primary transition-colors">
                    {card.title}
                  </h3>
                  <p className="mt-2 text-xs text-muted-foreground leading-relaxed">
                    {card.desc}
                  </p>
                </div>

                {/* Graphic Chart representation */}
                <div className="bg-slate-900 border-t border-border aspect-[4/3] relative flex items-center justify-center overflow-hidden p-2">
                  <img
                    src={card.image}
                    alt={`${card.title} Metrics`}
                    className="max-w-full max-h-full object-contain group-hover:scale-102 transition-transform duration-300"
                    loading="lazy"
                  />
                </div>

                {/* Caption Description */}
                <div className="p-4 bg-muted/30 border-t border-border text-[11px] text-muted-foreground leading-normal italic">
                  {card.caption}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* CTA section */}
        <section className="mt-20 relative overflow-hidden rounded-3xl bg-surface border border-border px-6 py-16 text-center sm:px-16 sm:py-20 shadow-sm">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0">
            <div className="h-64 w-64 rounded-full bg-primary/5 blur-3xl"></div>
          </div>
          <div className="relative z-10 mx-auto max-w-md">
            <h2 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
              Ready to analyze a study?
            </h2>
            <p className="mt-3 text-xs sm:text-sm text-muted-foreground leading-relaxed">
              Upload a chest CT scan or chest X-ray. Get structured summaries, pathology detections, and translation outputs in seconds.
            </p>
            <div className="mt-8 flex justify-center gap-3">
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground shadow-md hover:bg-primary/95 transition-all duration-200 cursor-pointer"
              >
                Get Started
              </Link>
              <Link
                href="/how-it-works"
                className="inline-flex items-center justify-center rounded-full border border-border bg-background px-6 py-3 text-sm font-semibold text-foreground hover:bg-muted/50 transition-all duration-200 cursor-pointer"
              >
                How it works
              </Link>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
