"use client";

import React from "react";
import Link from "next/link";
import Header from "@/components/Header";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-[#f8fafc] font-sans antialiased text-slate-800">
      <Header activePage="/" />

      {/* Main Content Hero */}
      <main className="flex-1 w-full mx-auto px-3 sm:px-6 py-10 sm:py-16 flex flex-col justify-center items-center gap-8 sm:gap-12 max-w-6xl">
        <div className="text-center space-y-3 max-w-2xl px-1">
          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent leading-tight">
            Lumora AI Platform
          </h1>
          <p className="text-xs sm:text-sm md:text-base text-slate-500 font-medium leading-relaxed">
            An advanced clinical decision-support system analyzing chest X-rays and CT scans.
            Generate narrative radiology reports, extract multi-label pathologies, and translate
            complex clinical findings into patient-friendly language.
          </p>
        </div>

        {/* Portal Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 w-full">
          {/* Card 1: Clinical Workspace */}
          <Link
            href="/workspace"
            className="group relative bg-white border border-slate-200 hover:border-blue-400 hover:shadow-lg rounded-2xl p-5 sm:p-8 flex flex-col justify-between transition-all duration-300 transform hover:-translate-y-1 overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-28 h-28 bg-blue-50/50 rounded-bl-full -z-10 group-hover:scale-110 transition-transform duration-300" />
            <div className="space-y-3 sm:space-y-4">
              <div className="h-11 w-11 sm:h-12 sm:w-12 rounded-xl bg-blue-50 border border-blue-100 text-blue-600 flex items-center justify-center text-2xl group-hover:bg-blue-600 group-hover:text-white transition-all duration-300">
                🩻
              </div>
              <h2 className="text-base sm:text-lg font-bold text-slate-800 group-hover:text-blue-600 transition-colors">
                Clinical Workspace
              </h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                Ingest medical images under strict modality guardrails, autogenerate draft radiology
                findings, detect pathologies via ClinicalBERT, and produce patient translations.
              </p>
            </div>
            <div className="mt-6 sm:mt-8 flex items-center gap-1 text-xs font-bold text-blue-600 group-hover:translate-x-1 transition-transform">
              Open Workspace <span>→</span>
            </div>
          </Link>

          {/* Card 2: Model Specifications */}
          <Link
            href="/models"
            className="group relative bg-white border border-slate-200 hover:border-indigo-400 hover:shadow-lg rounded-2xl p-5 sm:p-8 flex flex-col justify-between transition-all duration-300 transform hover:-translate-y-1 overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-28 h-28 bg-indigo-50/50 rounded-bl-full -z-10 group-hover:scale-110 transition-transform duration-300" />
            <div className="space-y-3 sm:space-y-4">
              <div className="h-11 w-11 sm:h-12 sm:w-12 rounded-xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center text-2xl group-hover:bg-indigo-600 group-hover:text-white transition-all duration-300">
                📈
              </div>
              <h2 className="text-base sm:text-lg font-bold text-slate-800 group-hover:text-indigo-600 transition-colors">
                Model Specs &amp; Analytics
              </h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                Explore individual model hyperparameter specifications, base architectures,
                datasets, training and validation splits, and learning curves.
              </p>
            </div>
            <div className="mt-6 sm:mt-8 flex items-center gap-1 text-xs font-bold text-indigo-600 group-hover:translate-x-1 transition-transform">
              View Specifications <span>→</span>
            </div>
          </Link>

          {/* Card 3: Literature Comparison */}
          <Link
            href="/comparison"
            className="group relative bg-white border border-slate-200 hover:border-emerald-400 hover:shadow-lg rounded-2xl p-5 sm:p-8 flex flex-col justify-between transition-all duration-300 transform hover:-translate-y-1 overflow-hidden sm:col-span-2 lg:col-span-1"
          >
            <div className="absolute top-0 right-0 w-28 h-28 bg-emerald-50/50 rounded-bl-full -z-10 group-hover:scale-110 transition-transform duration-300" />
            <div className="space-y-3 sm:space-y-4">
              <div className="h-11 w-11 sm:h-12 sm:w-12 rounded-xl bg-emerald-50 border border-emerald-100 text-emerald-600 flex items-center justify-center text-2xl group-hover:bg-emerald-600 group-hover:text-white transition-all duration-300">
                📊
              </div>
              <h2 className="text-base sm:text-lg font-bold text-slate-800 group-hover:text-emerald-600 transition-colors">
                Literature Comparison
              </h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                Compare Lumora with published medical SOTA models. View feature matrices, modality
                scope, and interactive pipeline simulation analysis.
              </p>
            </div>
            <div className="mt-6 sm:mt-8 flex items-center gap-1 text-xs font-bold text-emerald-600 group-hover:translate-x-1 transition-transform">
              View Comparison <span>→</span>
            </div>
          </Link>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-5 mt-8 text-center text-[10px] font-medium tracking-wide text-slate-400 px-3">
        <p>LUMORA CLINICAL DECISION-SUPPORT SYSTEM — DESIGNED FOR LICENSED HEALTHCARE PROVIDERS</p>
      </footer>
    </div>
  );
}
