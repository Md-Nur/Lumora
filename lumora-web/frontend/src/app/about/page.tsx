"use client";

import React from "react";
import Link from "next/link";

const TEAM_MEMBERS = [
  {
    name: "Md. Nur E Alam Siddiquee",
    role: "Co-Developer & Researcher",
    dept: "Electrical and Electronic Engineering (EEE)",
    univ: "University of Rajshahi",
    initials: "NS",
    gradient: "from-blue-600 to-cyan-500",
  },
  {
    name: "Mozammel Haque",
    role: "Co-Developer & Researcher",
    dept: "Electrical and Electronic Engineering (EEE)",
    univ: "University of Rajshahi",
    initials: "MH",
    gradient: "from-indigo-600 to-blue-500",
  },
  {
    name: "Pranto Protim Dutta",
    role: "Co-Developer & Researcher",
    dept: "Electrical and Electronic Engineering (EEE)",
    univ: "University of Rajshahi",
    initials: "PD",
    gradient: "from-purple-600 to-indigo-500",
  },
];

export default function About() {
  return (
    <div className="flex flex-col bg-background min-h-screen">
      <main className="mx-auto max-w-5xl px-6 py-20 w-full">
        {/* Hero Section */}
        <div className="text-center md:text-left mb-16">
          <p className="text-xs uppercase tracking-[0.2em] text-primary-deep font-semibold">
            Our Story
          </p>
          <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl text-foreground">
            Meet the team behind Lumora.
          </h1>
          <p className="mt-4 max-w-2xl text-muted-foreground leading-relaxed">
            This project is created by three passionate engineering students from the University of Rajshahi. We aim to combine advanced artificial intelligence with intuitive design to make medical scans easier to read.
          </p>
        </div>

        {/* Team Grid */}
        <section className="grid gap-6 sm:grid-cols-1 md:grid-cols-3 border-t border-border/60 py-16">
          {TEAM_MEMBERS.map((member, idx) => (
            <div
              key={idx}
              className="group flex flex-col items-center text-center p-8 rounded-3xl border border-border bg-card shadow-xs hover:shadow-md hover:border-primary/40 transition-all duration-300"
            >
              {/* Initials Avatar */}
              <div className={`flex h-16 w-16 items-center justify-center rounded-2xl bg-gradient-to-br ${member.gradient} text-white font-bold text-lg shadow-sm group-hover:scale-105 transition-transform duration-300 mb-6`}>
                {member.initials}
              </div>
              
              <h3 className="text-lg font-bold text-foreground group-hover:text-primary transition-colors">
                {member.name}
              </h3>
              
              <p className="text-xs text-primary-deep font-semibold mt-1">
                {member.role}
              </p>

              <div className="mt-6 border-t border-border/50 pt-4 w-full text-xs text-muted-foreground leading-relaxed">
                <p className="font-medium text-slate-600">{member.dept}</p>
                <p className="mt-0.5">{member.univ}</p>
              </div>
            </div>
          ))}
        </section>

        {/* Institution Context */}
        <section className="grid gap-12 sm:grid-cols-1 md:grid-cols-2 border-t border-border/60 py-16">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">
              University of Rajshahi
            </h2>
            <p className="text-xs text-primary-deep uppercase tracking-widest font-semibold mt-1">
              EEE Department
            </p>
            <p className="mt-4 text-sm text-muted-foreground leading-relaxed">
              Electrical and Electronic Engineering (EEE) provides the foundational pillars for signal processing, medical instrumentation, and computational intelligence. 
            </p>
            <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
              Combining our engineering curriculum with modern deep learning frameworks has allowed us to tackle challenges in computational medicine and clinical translation.
            </p>
          </div>
          <div className="bg-card rounded-3xl border border-border p-8 flex flex-col justify-center">
            <h3 className="text-lg font-bold text-foreground mb-4">Academic Focus</h3>
            <p className="text-sm text-muted-foreground leading-relaxed">
              Our research is supported by the academic excellence of the University of Rajshahi, one of the premier public research universities in Bangladesh. The EEE department fosters innovation in hardware design, machine learning, and biometric diagnostics.
            </p>
          </div>
        </section>

        {/* Future Vision Section */}
        <section className="border-t border-border/60 py-16">
          <div className="max-w-3xl">
            <h2 className="text-2xl font-bold tracking-tight text-foreground">
              Our Future Plan: Clinical Software Integration
            </h2>
            <p className="mt-4 text-sm text-muted-foreground leading-relaxed">
              Lumora was designed not just as a standalone research web application, but as a framework capable of embedding directly within real-world clinical systems.
            </p>
            <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
              Our ultimate objective is to integrate this AI assistant directly into actual physical X-ray machines and CT scanner software suites. By embedding our diagnostic classifiers, modality guardrails, and patient-friendly translation models into PACS (Picture Archiving and Communication Systems) and scanner console computers, we hope to provide instant, real-time diagnostic decision support at the point of care in hospitals.
            </p>
          </div>
        </section>

        {/* CTA section */}
        <section className="mt-8 relative overflow-hidden rounded-3xl bg-surface border border-border px-6 py-16 text-center sm:px-16 sm:py-20 shadow-sm">
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0">
            <div className="h-64 w-64 rounded-full bg-primary/5 blur-3xl"></div>
          </div>
          <div className="relative z-10 mx-auto max-w-md">
            <h2 className="text-2xl font-bold tracking-tight text-foreground sm:text-3xl">
              Explore Lumora
            </h2>
            <p className="mt-3 text-xs sm:text-sm text-muted-foreground leading-relaxed">
              Try analyzing a study using one of our sample scans or upload your own diagnostic image.
            </p>
            <div className="mt-8 flex justify-center gap-3">
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center rounded-full bg-primary px-8 py-3 text-sm font-semibold text-primary-foreground shadow-md hover:bg-primary/95 transition-all duration-200 cursor-pointer"
              >
                Go to Workspace
              </Link>
              <Link
                href="/how-it-works"
                className="inline-flex items-center justify-center rounded-full border border-border bg-background px-6 py-3 text-sm font-semibold text-foreground hover:bg-muted/50 transition-all duration-200 cursor-pointer"
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
