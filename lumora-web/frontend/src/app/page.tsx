"use client";

import React from "react";
import Link from "next/link";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-[#f8fafc] font-sans antialiased text-slate-800">
      {/* Top Premium Nav Header */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md sticky top-0 z-50 shadow-xs">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="flex items-center h-12">
              <img
                src="/logo.png"
                alt="Lumora Logo"
                className="h-11 w-auto object-contain filter hover:brightness-105 transition-all"
              />
            </div>
            <div className="h-6 w-[1px] bg-slate-200 mx-1"></div>
            <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5 border border-slate-200">
              <Link
                href="/"
                className="px-3 py-1.5 rounded-md text-xs font-bold bg-white text-blue-600 shadow-xs border border-blue-100 transition-all"
              >
                Portal Home
              </Link>
              <Link
                href="/workspace"
                className="px-3 py-1.5 rounded-md text-xs font-bold text-slate-500 hover:text-slate-700 transition-all"
              >
                Clinical Workspace
              </Link>
              <Link
                href="/models"
                className="px-3 py-1.5 rounded-md text-xs font-bold text-slate-500 hover:text-slate-700 transition-all"
              >
                Model Specifications
              </Link>
              <Link
                href="/comparison"
                className="px-3 py-1.5 rounded-md text-xs font-bold text-slate-500 hover:text-slate-700 transition-all"
              >
                Literature Comparison
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Hero */}
      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-16 flex flex-col justify-center items-center gap-12">
        <div className="text-center space-y-4 max-w-2xl">
          <h1 className="text-4xl md:text-5xl font-extrabold tracking-tight bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
            Lumora AI Platform
          </h1>
          <p className="text-sm md:text-base text-slate-500 font-medium leading-relaxed">
            An advanced clinical decision-support system analyzing chest X-rays and CT scans. 
            Generate narrative radiology reports, extract multi-label pathologies, and translate complex clinical findings into patient-friendly language.
          </p>
        </div>

        {/* Portal Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full">
          {/* Card 1: Clinical Workspace */}
          <Link
            href="/workspace"
            className="group relative bg-white border border-slate-200 hover:border-blue-400 hover:shadow-lg rounded-2xl p-8 flex flex-col justify-between transition-all duration-300 transform hover:-translate-y-1 overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-blue-50/50 rounded-bl-full -z-10 group-hover:scale-110 transition-transform duration-300" />
            <div className="space-y-4">
              <div className="h-12 w-12 rounded-xl bg-blue-50 border border-blue-100 text-blue-600 flex items-center justify-center text-2xl group-hover:bg-blue-600 group-hover:text-white transition-all duration-300">
                🩻
              </div>
              <h2 className="text-lg font-bold text-slate-800 group-hover:text-blue-600 transition-colors">
                Clinical Workspace
              </h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                Ingest medical images under strict modality guardrails. autogenerate draft radiology findings, detect pathologies via ClinicalBERT, and produce patient layperson translations.
              </p>
            </div>
            <div className="mt-8 flex items-center gap-1 text-xs font-bold text-blue-600 group-hover:translate-x-1 transition-transform">
              Open Workspace <span>→</span>
            </div>
          </Link>

          {/* Card 2: Model Specifications & Performance */}
          <Link
            href="/models"
            className="group relative bg-white border border-slate-200 hover:border-indigo-400 hover:shadow-lg rounded-2xl p-8 flex flex-col justify-between transition-all duration-300 transform hover:-translate-y-1 overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-50/50 rounded-bl-full -z-10 group-hover:scale-110 transition-transform duration-300" />
            <div className="space-y-4">
              <div className="h-12 w-12 rounded-xl bg-indigo-50 border border-indigo-100 text-indigo-600 flex items-center justify-center text-2xl group-hover:bg-indigo-600 group-hover:text-white transition-all duration-300">
                📈
              </div>
              <h2 className="text-lg font-bold text-slate-800 group-hover:text-indigo-600 transition-colors">
                Model Specs & Analytics
              </h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                Explore individual model hyperparameter specifications, base architectures, datasets (including training, validation, and test set splits), and training curves.
              </p>
            </div>
            <div className="mt-8 flex items-center gap-1 text-xs font-bold text-indigo-600 group-hover:translate-x-1 transition-transform">
              View Specifications <span>→</span>
            </div>
          </Link>

          {/* Card 3: Literature Comparison */}
          <Link
            href="/comparison"
            className="group relative bg-white border border-slate-200 hover:border-emerald-400 hover:shadow-lg rounded-2xl p-8 flex flex-col justify-between transition-all duration-300 transform hover:-translate-y-1 overflow-hidden"
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-emerald-50/50 rounded-bl-full -z-10 group-hover:scale-110 transition-transform duration-300" />
            <div className="space-y-4">
              <div className="h-12 w-12 rounded-xl bg-emerald-50 border border-emerald-100 text-emerald-600 flex items-center justify-center text-2xl group-hover:bg-emerald-600 group-hover:text-white transition-all duration-300">
                📊
              </div>
              <h2 className="text-lg font-bold text-slate-800 group-hover:text-emerald-600 transition-colors">
                Literature Comparison
              </h2>
              <p className="text-xs text-slate-500 leading-relaxed">
                Compare Lumora with published medical SOTA models. View feature matrices, modalities scope, and interactive analysis.
              </p>
            </div>
            <div className="mt-8 flex items-center gap-1 text-xs font-bold text-emerald-600 group-hover:translate-x-1 transition-transform">
              View Comparison <span>→</span>
            </div>
          </Link>
        </div>
      </main>

      {/* Clean clinical footer */}
      <footer className="border-t border-slate-200 bg-white py-6 mt-12 text-center text-[10px] font-medium tracking-wide text-slate-400">
        <p>
          LUMORA CLINICAL DECISION-SUPPORT SYSTEM — DESIGNED FOR LICENSED
          HEALTHCARE PROVIDERS
        </p>
      </footer>
    </div>
  );
}
