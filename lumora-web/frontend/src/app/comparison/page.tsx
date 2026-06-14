"use client";

import React, { useState } from "react";
import Link from "next/link";

interface FeatureRow {
  name: string;
  category: "capability" | "architecture" | "clinical";
  lumora: { value: string; check: boolean; detail: string };
  alshmrani: { value: string; check: boolean; detail: string };
  sharma: { value: string; check: boolean; detail: string };
}

const COMPARISON_FEATURES: FeatureRow[] = [
  {
    name: "Supported Modalities",
    category: "capability",
    lumora: { value: "X-Ray & CT Scan", check: true, detail: "Dual modality: handles standard 2D X-rays and converted 2D slices of axial 3D CT volumes." },
    alshmrani: { value: "X-Ray Only", check: false, detail: "Strictly limited to chest radiograph (CXR) imagery." },
    sharma: { value: "X-Ray Only", check: false, detail: "Strictly limited to chest radiograph (CXR) imagery." }
  },
  {
    name: "Supported Disease Classes",
    category: "clinical",
    lumora: { value: "26 Pathologies", check: true, detail: "Comprehensive vocabulary across both cardiopulmonary and extra-pulmonary findings." },
    alshmrani: { value: "6 Classes", check: false, detail: "COVID-19, Normal, Pneumonia, Lung Opacity, Tuberculosis, and Lung Cancer." },
    sharma: { value: "2 Classes", check: false, detail: "Binary classification only (Pneumonia vs. Normal)." }
  },
  {
    name: "Narrative Report Gen",
    category: "capability",
    lumora: { value: "Fully Generative", check: true, detail: "Autoregressively drafts complete clinical findings using a visual projection link to GPT-2." },
    alshmrani: { value: "No (Labels Only)", check: false, detail: "Outputs only isolated class classification categories." },
    sharma: { value: "No (Labels Only)", check: false, detail: "Outputs only binary prediction flags." }
  },
  {
    name: "Layperson Translation",
    category: "clinical",
    lumora: { value: "T5 + LoRA Adapter", check: true, detail: "Automatically translates clinical report jargon into clear, patient-friendly lay terms." },
    alshmrani: { value: "None", check: false, detail: "Does not translate or produce natural text output." },
    sharma: { value: "None", check: false, detail: "Does not translate or produce natural text output." }
  },
  {
    name: "Modality Guardrail",
    category: "architecture",
    lumora: { value: "YOLOv11s Classifier", check: true, detail: "Validates incoming scan modality and filters non-medical images to save compute." },
    alshmrani: { value: "None", check: false, detail: "Assumes uploaded images are pre-cleared chest X-rays." },
    sharma: { value: "None", check: false, detail: "Assumes uploaded images are pre-cleared chest X-rays." }
  },
  {
    name: "Core Architecture Type",
    category: "architecture",
    lumora: { value: "5-Model Pipeline", check: true, detail: "Modular pipeline combining classification, autoregressive VLM generation, ClinicalBERT, and Seq2Seq translation." },
    alshmrani: { value: "VGG19 + Custom CNN", check: false, detail: "Monolithic transfer learning network for classification." },
    sharma: { value: "VGG-16 + MLP Classifier", check: false, detail: "Monolithic transfer learning network for classification." }
  },
  {
    name: "Co-occurring Findings",
    category: "clinical",
    lumora: { value: "Multi-Label Extraction", check: true, detail: "Can identify any combination of the 26 pathologies present concurrently." },
    alshmrani: { value: "Single-Class Focus", check: false, detail: "Softmax output prioritizes a single dominant disease label." },
    sharma: { value: "Binary Focus", check: false, detail: "Outputs only a single prediction probability for pneumonia." }
  }
];

const CASE_SIMULATOR_SCENARIOS = [
  {
    id: "pneumonia-effusion",
    title: "Scenario A: X-Ray with Pneumonia & Pleural Effusion",
    description: "An X-ray showing consolidation indicative of pneumonia, along with fluid in the pleural cavity.",
    lumoraSteps: [
      { step: "1. YOLO Guardrail", status: "Verified X-Ray (99.8%)", color: "emerald" },
      { step: "2. Report Generator", status: "Generated: 'Findings show patchy consolidation and moderate pleural effusion.'", color: "blue" },
      { step: "3. ClinicalBERT Classifier", status: "Extracted: [Pneumonia: 0.82, Pleural Effusion: 0.76]", color: "rose" },
      { step: "4. T5 Translator", status: "Translated: 'You have signs of pneumonia (lung infection) and some fluid accumulation around the lungs.'", color: "amber" }
    ],
    alshmraniOutput: "Outputs single classification: 'Pneumonia' (fails to flag pleural effusion due to single-label classification limitation).",
    sharmaOutput: "Outputs binary label: 'Pneumonia' (fails to flag pleural effusion; does not generate clinical details)."
  },
  {
    id: "ct-hernia",
    title: "Scenario B: CT Scan with Hiatal Hernia",
    description: "A representative 2D axial CT slice capturing the stomach herniating through the diaphragm.",
    lumoraSteps: [
      { step: "1. YOLO Guardrail", status: "Verified CT Scan (99.4%)", color: "emerald" },
      { step: "2. Report Generator", status: "Generated: 'Visualized herniation of the gastric cardia through the esophageal hiatus.'", color: "blue" },
      { step: "3. ClinicalBERT Classifier", status: "Extracted: [Hiatal Hernia: 0.94]", color: "rose" },
      { step: "4. T5 Translator", status: "Translated: 'The top part of your stomach is sliding up into your chest cavity.'", color: "amber" }
    ],
    alshmraniOutput: "Rejects or processes incorrectly: Model is only trained on chest X-rays. Output is undefined or returns random false labels.",
    sharmaOutput: "Rejects or processes incorrectly: Model is only trained on chest X-rays. Returns standard 'Normal' or false 'Pneumonia' predictions."
  },
  {
    id: "invalid-image",
    title: "Scenario C: Non-Medical Image Upload (Selfie)",
    description: "A patient accidentally uploads a portrait photograph instead of a scan.",
    lumoraSteps: [
      { step: "1. YOLO Guardrail", status: "REJECTED (Modality: Other, Confidence: 99.9%)", color: "rose" },
      { step: "2. Report Generator", status: "Bypassed / Not Executed (Compute saved)", color: "slate" },
      { step: "3. ClinicalBERT Classifier", status: "Bypassed / Not Executed", color: "slate" },
      { step: "4. T5 Translator", status: "Bypassed / Not Executed", color: "slate" }
    ],
    alshmraniOutput: "Attempts forward pass on selfie. Incorrectly classifies as 'Normal' or 'Lung Cancer' with high confidence.",
    sharmaOutput: "Attempts forward pass on selfie. Incorrectly predicts 'Pneumonia' or 'Normal' based on noise features."
  }
];

export default function LiteratureComparison() {
  const [activeTab, setActiveTab] = useState<"all" | "capability" | "architecture" | "clinical">("all");
  const [hoveredFeature, setHoveredFeature] = useState<string | null>(null);
  const [activeScenario, setActiveScenario] = useState<string>("pneumonia-effusion");

  const filteredFeatures = activeTab === "all" 
    ? COMPARISON_FEATURES 
    : COMPARISON_FEATURES.filter(f => f.category === activeTab);

  const scenario = CASE_SIMULATOR_SCENARIOS.find(s => s.id === activeScenario) || CASE_SIMULATOR_SCENARIOS[0];

  return (
    <div className="flex flex-col min-h-screen bg-[#f8fafc] font-sans antialiased text-slate-800">
      {/* Navigation Header */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md sticky top-0 z-50 shadow-xs">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center h-12">
              <img
                src="/logo.png"
                alt="Lumora Logo"
                className="h-11 w-auto object-contain filter hover:brightness-105 transition-all"
              />
            </Link>
            <div className="h-6 w-[1px] bg-slate-200 mx-1"></div>
            <div className="flex items-center gap-1 bg-slate-100 rounded-lg p-0.5 border border-slate-200">
              <Link
                href="/"
                className="px-3 py-1.5 rounded-md text-xs font-bold text-slate-500 hover:text-slate-700 transition-all"
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
                className="px-3 py-1.5 rounded-md text-xs font-bold bg-white text-blue-600 shadow-xs border border-blue-100 transition-all"
              >
                Literature Comparison
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-6 space-y-8">
        
        {/* Page Hero Header */}
        <div className="text-center space-y-3 py-6 max-w-3xl mx-auto">
          <span className="text-[10px] uppercase font-bold tracking-widest bg-blue-100 text-blue-800 px-3 py-1 rounded-full border border-blue-200">
            Literature Comparison
          </span>
          <h1 className="text-3xl md:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
            Scientific Advancements & Benchmark Analysis
          </h1>
          <p className="text-xs md:text-sm text-slate-500 font-medium leading-relaxed">
            See how Lumora's modular 5-model pipeline stacks up against existing models in published research literature, moving beyond classification labels to a fully unified, multi-modal clinical workspace.
          </p>
        </div>

        {/* Counter Analytics Cards */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="clinical-card p-5 border-l-4 border-blue-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Modality Scope</div>
            <div className="text-2xl font-black text-slate-800 mt-1">2 <span className="text-sm font-semibold text-slate-500">Supported</span></div>
            <div className="text-[10px] text-slate-500 mt-1">X-Ray + CT volumes (Others: CXR only)</div>
          </div>
          <div className="clinical-card p-5 border-l-4 border-emerald-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Clinical Pipeline</div>
            <div className="text-2xl font-black text-slate-800 mt-1">5 <span className="text-sm font-semibold text-slate-500">Models</span></div>
            <div className="text-[10px] text-slate-500 mt-1">End-to-end processing pipeline</div>
          </div>
          <div className="clinical-card p-5 border-l-4 border-rose-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Disease Mapping</div>
            <div className="text-2xl font-black text-slate-800 mt-1">26 <span className="text-sm font-semibold text-slate-500">Classes</span></div>
            <div className="text-[10px] text-slate-500 mt-1">Multi-label classification (Others: 2 to 6)</div>
          </div>
          <div className="clinical-card p-5 border-l-4 border-amber-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Patient Care</div>
            <div className="text-2xl font-black text-slate-800 mt-1">Yes <span className="text-sm font-semibold text-slate-500">Layperson</span></div>
            <div className="text-[10px] text-slate-500 mt-1">T5 adapter translations for patient clarity</div>
          </div>
        </div>

        {/* Feature Matrix Header / Filters */}
        <div className="clinical-card p-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b border-slate-100 pb-4 mb-4 gap-3">
            <div>
              <h2 className="text-sm font-bold text-slate-800 tracking-wide uppercase font-mono">
                📋 Interactive Feature Comparison Matrix
              </h2>
              <p className="text-[10px] text-slate-400 mt-1">Click categories below to filter dimensions</p>
            </div>
            
            {/* Filter buttons */}
            <div className="flex flex-wrap gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200">
              {(["all", "capability", "clinical", "architecture"] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`px-3 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${
                    activeTab === tab
                      ? "bg-white text-slate-800 shadow-xs border border-slate-200"
                      : "text-slate-500 hover:text-slate-700"
                  }`}
                >
                  {tab}
                </button>
              ))}
            </div>
          </div>

          {/* Interactive Feature Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-400">Feature</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-blue-700 bg-blue-50/50 rounded-t-lg">Lumora (Ours)</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-500">Alshmrani et al. (2023)</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-500">Sharma & Guleria (2023)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {filteredFeatures.map((row) => (
                  <tr 
                    key={row.name} 
                    className={`hover:bg-slate-50/70 transition-colors cursor-pointer ${hoveredFeature === row.name ? "bg-slate-50" : ""}`}
                    onClick={() => setHoveredFeature(hoveredFeature === row.name ? null : row.name)}
                  >
                    <td className="py-4 px-4">
                      <div className="text-xs font-bold text-slate-700">{row.name}</div>
                      <div className="text-[10px] uppercase font-mono font-semibold text-slate-400 mt-0.5">{row.category}</div>
                    </td>
                    
                    {/* Lumora */}
                    <td className="py-4 px-4 bg-blue-50/20 font-medium border-x border-blue-50">
                      <div className="flex items-center gap-1.5">
                        <span className="text-emerald-500 text-xs">✅</span>
                        <span className="text-xs font-semibold text-blue-900">{row.lumora.value}</span>
                      </div>
                      {hoveredFeature === row.name && (
                        <div className="text-[10px] text-blue-700 mt-1 max-w-[280px] leading-relaxed font-medium bg-blue-100/50 p-2 rounded-lg border border-blue-100">
                          {row.lumora.detail}
                        </div>
                      )}
                    </td>
                    
                    {/* Alshmrani */}
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-1.5">
                        <span className={row.alshmrani.check ? "text-emerald-500 text-xs" : "text-rose-400 text-xs"}>
                          {row.alshmrani.check ? "✅" : "❌"}
                        </span>
                        <span className="text-xs font-semibold text-slate-600">{row.alshmrani.value}</span>
                      </div>
                      {hoveredFeature === row.name && (
                        <div className="text-[10px] text-slate-500 mt-1 max-w-[280px] leading-relaxed">
                          {row.alshmrani.detail}
                        </div>
                      )}
                    </td>
                    
                    {/* Sharma */}
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-1.5">
                        <span className={row.sharma.check ? "text-emerald-500 text-xs" : "text-rose-400 text-xs"}>
                          {row.sharma.check ? "✅" : "❌"}
                        </span>
                        <span className="text-xs font-semibold text-slate-600">{row.sharma.value}</span>
                      </div>
                      {hoveredFeature === row.name && (
                        <div className="text-[10px] text-slate-500 mt-1 max-w-[280px] leading-relaxed">
                          {row.sharma.detail}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="text-[10px] text-slate-400 text-center mt-3 pt-3 border-t border-slate-50 italic">
            * Click any row to expand detailed architectural specifications and rationale.
          </div>
        </div>

        {/* Interactive Case Simulator */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="clinical-card p-6 lg:col-span-1 bg-slate-900 border-slate-800 text-white animate-fade-in">
            <h3 className="text-xs font-bold text-blue-400 tracking-wide uppercase font-mono border-b border-slate-800 pb-2 mb-4">
              🧪 Pipeline Case Simulator
            </h3>
            <p className="text-[11px] text-slate-400 leading-relaxed mb-4">
              Select a simulated diagnostic scenario below to visualize and trace the sequential pipeline processing differences.
            </p>
            
            {/* Scenario Buttons */}
            <div className="flex flex-col gap-2">
              {CASE_SIMULATOR_SCENARIOS.map(sc => (
                <button
                  key={sc.id}
                  onClick={() => setActiveScenario(sc.id)}
                  className={`w-full text-left p-3 rounded-xl border text-xs font-bold transition-all ${
                    activeScenario === sc.id
                      ? "bg-blue-600 border-blue-500 text-white shadow-md"
                      : "bg-slate-800/50 border-slate-700 text-slate-300 hover:bg-slate-800 hover:text-white"
                  }`}
                >
                  <div className="truncate">{sc.title}</div>
                  <div className="text-[9px] font-normal text-slate-400 mt-1 line-clamp-1">{sc.description}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Simulator Details */}
          <div className="clinical-card p-6 lg:col-span-2 bg-white flex flex-col justify-between">
            <div className="space-y-4">
              <div>
                <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2">
                  ⚡ Pipeline Execution Trace
                </h3>
                <p className="text-[11px] text-slate-400 mt-1 italic">{scenario.description}</p>
              </div>

              {/* Steps grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {/* Lumora execution */}
                <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-4 space-y-3">
                  <div className="text-[10px] font-bold text-blue-700 uppercase tracking-wider">Lumora (Ours) Trace</div>
                  <div className="space-y-2">
                    {scenario.lumoraSteps.map((st, idx) => (
                      <div key={idx} className="bg-white rounded-lg p-2 border border-slate-100 shadow-2xs">
                        <div className="text-[9px] font-mono font-bold text-slate-400">{st.step}</div>
                        <div className="text-[10px] font-semibold text-slate-800 mt-0.5 leading-snug">{st.status}</div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Comparative Models execution */}
                <div className="flex flex-col gap-3">
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 flex-1">
                    <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">Alshmrani et al. (VGG19)</div>
                    <div className="text-[11px] font-medium text-slate-700 mt-2 leading-relaxed bg-white border border-slate-100 rounded-lg p-2 shadow-2xs">
                      {scenario.alshmraniOutput}
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-4 flex-1">
                    <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">Sharma & Guleria (VGG-16)</div>
                    <div className="text-[11px] font-medium text-slate-700 mt-2 leading-relaxed bg-white border border-slate-100 rounded-lg p-2 shadow-2xs">
                      {scenario.sharmaOutput}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="text-[10px] text-slate-400 mt-4 pt-3 border-t border-slate-100 flex justify-between items-center">
              <span>Simulation mode: Sandbox (Offline API)</span>
              <span className="font-mono bg-slate-100 px-1.5 py-0.5 rounded text-[9px] text-slate-500">Pipeline OK</span>
            </div>
          </div>
        </div>

        {/* Detailed Literature Summaries */}
        <div>
          <h2 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono mb-4">
            📚 Comparative Literature Citations & Methodology
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            {/* Paper 1 details */}
            <div className="clinical-card p-5">
              <span className="text-[9px] font-mono font-bold bg-slate-100 text-slate-600 px-2 py-0.5 rounded">Reference 01</span>
              <h4 className="text-xs font-bold text-slate-800 mt-2 leading-snug">
                A deep learning architecture for multi-class lung diseases classification using chest X-ray (CXR) images
              </h4>
              <p className="text-[10px] font-medium text-slate-400 mt-1 italic">
                Alexandria Engineering Journal (2023)
              </p>
              
              <div className="mt-4 space-y-2 text-[11px] text-slate-600">
                <p>
                  <strong>Methodology Summary:</strong> Utilized a pre-trained VGG19 network combined with three custom CNN feature extraction blocks. Softmax output for 6 single-class lung diseases.
                </p>
                <div className="bg-slate-50 p-2 rounded-lg border border-slate-100 font-mono text-[10px] text-slate-500">
                  Accuracy: 96.48% | Recall: 93.75% | Precision: 97.56% | F1: 95.62%
                </div>
                <p className="text-[10px] text-slate-500">
                  <strong>Key Limitations:</strong> Lacks text report generation capabilities, is completely restricted to Chest X-ray modalities, and fails to model multi-label clinical diagnoses.
                </p>
              </div>
              
              <div className="mt-4 pt-3 border-t border-slate-100">
                <a 
                  href="https://doi.org/10.1016/j.aej.2022.10.053" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-[10px] font-bold text-blue-600 hover:underline inline-flex items-center gap-1"
                >
                  View Paper via DOI <span>→</span>
                </a>
              </div>
            </div>

            {/* Paper 2 details */}
            <div className="clinical-card p-5">
              <span className="text-[9px] font-mono font-bold bg-slate-100 text-slate-600 px-2 py-0.5 rounded">Reference 02</span>
              <h4 className="text-xs font-bold text-slate-800 mt-2 leading-snug">
                A Deep Learning based model for the Detection of Pneumonia from Chest X-Ray Images using VGG-16 and Neural Networks
              </h4>
              <p className="text-[10px] font-medium text-slate-400 mt-1 italic">
                Procedia Computer Science (2023)
              </p>
              
              <div className="mt-4 space-y-2 text-[11px] text-slate-600">
                <p>
                  <strong>Methodology Summary:</strong> Employs transfer learning based fine-tuning of a VGG-16 backbone with custom dense layers to binary classify radiographs.
                </p>
                <div className="bg-slate-50 p-2 rounded-lg border border-slate-100 font-mono text-[10px] text-slate-500">
                  Binary Accuracy: ~90.00% to 92.50% (Dataset Dependent)
                </div>
                <p className="text-[10px] text-slate-500">
                  <strong>Key Limitations:</strong> Extremely narrow task scope (binary pneumonia classification only); fails on non-pneumonic findings, and does not produce clinical documentation or patient translations.
                </p>
              </div>
              
              <div className="mt-4 pt-3 border-t border-slate-100">
                <a 
                  href="https://doi.org/10.1016/j.procs.2023.01.018" 
                  target="_blank" 
                  rel="noopener noreferrer"
                  className="text-[10px] font-bold text-blue-600 hover:underline inline-flex items-center gap-1"
                >
                  View Paper via DOI <span>→</span>
                </a>
              </div>
            </div>

          </div>
        </div>

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 bg-white py-6 mt-12 text-center text-[10px] font-medium tracking-wide text-slate-400">
        <p>
          LUMORA CLINICAL DECISION-SUPPORT SYSTEM — DESIGNED FOR LICENSED
          HEALTHCARE PROVIDERS
        </p>
      </footer>
    </div>
  );
}
