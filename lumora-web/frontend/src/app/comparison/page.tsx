"use client";

import React, { useState } from "react";
import Header from "@/components/Header";

interface FeatureRow {
  name: string;
  category: "capability" | "architecture" | "clinical";
  lumora: { value: string; check: boolean; detail: string };
  alshmrani: { value: string; check: boolean; detail: string };
  sharma: { value: string; check: boolean; detail: string };
  hamamci: { value: string; check: boolean; detail: string };
  zhang: { value: string; check: boolean; detail: string };
}

const COMPARISON_FEATURES: FeatureRow[] = [
  {
    name: "Supported Modalities",
    category: "capability",
    lumora: { value: "X-Ray & CT Scan", check: true, detail: "Dual modality: handles standard 2D X-rays and converted 2D slices of axial 3D CT volumes." },
    alshmrani: { value: "X-Ray Only", check: false, detail: "Strictly limited to chest radiograph (CXR) imagery." },
    sharma: { value: "X-Ray Only", check: false, detail: "Strictly limited to chest radiograph (CXR) imagery." },
    hamamci: { value: "3D CT Volumes", check: true, detail: "Foundation models specifically trained on 3D chest CT volumes." },
    zhang: { value: "3D CT Volumes", check: true, detail: "Grounded VLM baselines trained on 3D chest CT volumes with region-level annotations." }
  },
  {
    name: "Supported Disease Classes",
    category: "clinical",
    lumora: { value: "26 Pathologies", check: true, detail: "Comprehensive vocabulary across both cardiopulmonary and extra-pulmonary findings." },
    alshmrani: { value: "6 Classes", check: false, detail: "COVID-19, Normal, Pneumonia, Lung Opacity, Tuberculosis, and Lung Cancer." },
    sharma: { value: "2 Classes", check: false, detail: "Binary classification only (Pneumonia vs. Normal)." },
    hamamci: { value: "18 CT Abnormalities", check: true, detail: "18 CT-specific abnormality labels from CT-RATE dataset." },
    zhang: { value: "Region-Grounded Labels", check: true, detail: "Region-grounded CT report supervision with anatomical segmentation labels." }
  },
  {
    name: "Narrative Report Gen",
    category: "capability",
    lumora: { value: "Fully Generative", check: true, detail: "Autoregressively drafts complete clinical findings using a visual projection link to GPT-2." },
    alshmrani: { value: "No (Labels Only)", check: false, detail: "Outputs only isolated class classification categories." },
    sharma: { value: "No (Labels Only)", check: false, detail: "Outputs only binary prediction flags." },
    hamamci: { value: "CT-CHAT Language", check: true, detail: "CT-CHAT foundation model supports CT-focused language interaction and report generation." },
    zhang: { value: "Grounded Reports", check: true, detail: "Provides grounded CT report-generation supervision with anatomical grounding." }
  },
  {
    name: "Layperson Translation",
    category: "clinical",
    lumora: { value: "T5 + LoRA Adapter", check: true, detail: "Automatically translates clinical report jargon into clear, patient-friendly lay terms." },
    alshmrani: { value: "None", check: false, detail: "Does not translate or produce natural text output." },
    sharma: { value: "None", check: false, detail: "Does not translate or produce natural text output." },
    hamamci: { value: "None", check: false, detail: "No dedicated layperson translation module." },
    zhang: { value: "None", check: false, detail: "No dedicated layperson translation module." }
  },
  {
    name: "Modality Guardrail",
    category: "architecture",
    lumora: { value: "YOLOv11s Classifier", check: true, detail: "Validates incoming scan modality and filters non-medical images to save compute." },
    alshmrani: { value: "None", check: false, detail: "Assumes uploaded images are pre-cleared chest X-rays." },
    sharma: { value: "None", check: false, detail: "Assumes uploaded images are pre-cleared chest X-rays." },
    hamamci: { value: "Not Front-Door", check: false, detail: "No input-screening pipeline; foundation model architecture." },
    zhang: { value: "Dataset Benchmark", check: false, detail: "Dataset benchmark tool, not an input-screening pipeline." }
  },
  {
    name: "Core Architecture Type",
    category: "architecture",
    lumora: { value: "5-Model Pipeline", check: true, detail: "Modular pipeline combining classification, autoregressive VLM generation, ClinicalBERT, and Seq2Seq translation." },
    alshmrani: { value: "VGG19 + Custom CNN", check: false, detail: "Monolithic transfer learning network for classification." },
    sharma: { value: "VGG-16 + MLP Classifier", check: false, detail: "Monolithic transfer learning network for classification." },
    hamamci: { value: "CT-CLIP + CT-CHAT", check: true, detail: "Foundation models built on multimodal CT understanding and language interaction." },
    zhang: { value: "Grounded VLM Baselines", check: true, detail: "Vision language models with region-level grounding and anatomical supervision." }
  },
  {
    name: "Co-occurring Findings",
    category: "clinical",
    lumora: { value: "Multi-Label Extraction", check: true, detail: "Can identify any combination of the 26 pathologies present concurrently." },
    alshmrani: { value: "Single-Class Focus", check: false, detail: "Softmax output prioritizes a single dominant disease label." },
    sharma: { value: "Binary Focus", check: false, detail: "Outputs only a single prediction probability for pneumonia." },
    hamamci: { value: "Multi-Abnormality", check: true, detail: "CT-RATE dataset supports multiple abnormalities per CT scan." },
    zhang: { value: "Multi-Finding VQA", check: true, detail: "Visual question answering pairs capture multiple anatomical findings." }
  }
];

interface CaseScenario {
  id: string;
  title: string;
  description: string;
  lumoraSteps: Array<{ step: string; status: string; color: string }>;
  alshmraniOutput: string;
  sharmaOutput: string;
  hamamciOutput: string;
  zhangOutput: string;
}

const CASE_SIMULATOR_SCENARIOS: CaseScenario[] = [
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
    sharmaOutput: "Outputs binary label: 'Pneumonia' (fails to flag pleural effusion; does not generate clinical details).",
    hamamciOutput: "Not applicable for X-ray (model is CT-specific). Would need CT scan input.",
    zhangOutput: "Not applicable for X-ray (model is CT-specific). Would need CT scan input."
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
    sharmaOutput: "Rejects or processes incorrectly: Model is only trained on chest X-rays. Returns standard 'Normal' or false 'Pneumonia' predictions.",
    hamamciOutput: "Processes 3D volume slices: CT-CHAT generates: 'Stomach herniation through diaphragm detected' (no multi-label extraction or patient translation).",
    zhangOutput: "Generates region-grounded report: 'Diaphragmatic hernia at region coordinates (x,y,z)' (provides localization but lacks patient-friendly translation)."
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
    sharmaOutput: "Attempts forward pass on selfie. Incorrectly predicts 'Pneumonia' or 'Normal' based on noise features.",
    hamamciOutput: "Attempts to process selfie as 3D data. Foundation model may hallucinate CT abnormalities due to lack of modality guardrail.",
    zhangOutput: "Generates spurious region-grounded annotations for facial features, misinterpreting them as anatomical structures."
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
      <Header activePage="/comparison" />

      {/* Main Content */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 py-4 sm:py-6 space-y-5 sm:space-y-8">
        
        {/* Page Hero Header */}
        <div className="text-center space-y-3 py-4 sm:py-6 max-w-3xl mx-auto">
          <span className="text-[10px] uppercase font-bold tracking-widest bg-blue-100 text-blue-800 px-3 py-1 rounded-full border border-blue-200">
            Literature Comparison
          </span>
          <h1 className="text-2xl sm:text-3xl md:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent leading-tight">
            Scientific Advancements & Benchmark Analysis
          </h1>
          <p className="text-xs md:text-sm text-slate-500 font-medium leading-relaxed">
            See how Lumora's modular 5-model pipeline stacks up against existing models in published research literature, moving beyond classification labels to a fully unified, multi-modal clinical workspace.
          </p>
        </div>

        {/* Counter Analytics Cards */}
        <div className="grid grid-cols-1 min-[360px]:grid-cols-2 md:grid-cols-4 gap-3 sm:gap-4">
          <div className="clinical-card p-3 sm:p-5 border-l-4 border-blue-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Modality Scope</div>
            <div className="text-2xl font-black text-slate-800 mt-1">2 <span className="text-sm font-semibold text-slate-500">Supported</span></div>
            <div className="text-[10px] text-slate-500 mt-1">X-Ray + CT volumes (Others: CXR only)</div>
          </div>
          <div className="clinical-card p-3 sm:p-5 border-l-4 border-emerald-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Clinical Pipeline</div>
            <div className="text-2xl font-black text-slate-800 mt-1">5 <span className="text-sm font-semibold text-slate-500">Models</span></div>
            <div className="text-[10px] text-slate-500 mt-1">End-to-end processing pipeline</div>
          </div>
          <div className="clinical-card p-3 sm:p-5 border-l-4 border-rose-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Disease Mapping</div>
            <div className="text-2xl font-black text-slate-800 mt-1">26 <span className="text-sm font-semibold text-slate-500">Classes</span></div>
            <div className="text-[10px] text-slate-500 mt-1">Multi-label classification (Others: 2 to 6)</div>
          </div>
          <div className="clinical-card p-3 sm:p-5 border-l-4 border-amber-500 hover:translate-y-[-2px] transition-all">
            <div className="text-[10px] font-mono text-slate-400 uppercase tracking-wider">Patient Care</div>
            <div className="text-2xl font-black text-slate-800 mt-1">Yes <span className="text-sm font-semibold text-slate-500">Layperson</span></div>
            <div className="text-[10px] text-slate-500 mt-1">T5 adapter translations for patient clarity</div>
          </div>
        </div>

        {/* Feature Matrix Header / Filters */}
        <div className="clinical-card p-3 sm:p-6">
          <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between border-b border-slate-100 pb-4 mb-4 gap-3">
            <div>
              <h2 className="text-sm font-bold text-slate-800 tracking-wide uppercase font-mono">
                📋 Interactive Feature Comparison Matrix
              </h2>
              <p className="text-[10px] text-slate-400 mt-1">Click categories below to filter dimensions</p>
            </div>
            
            {/* Filter buttons */}
            <div className="flex flex-wrap gap-1 bg-slate-100 p-0.5 rounded-lg border border-slate-200 w-full sm:w-auto">
              {(["all", "capability", "clinical", "architecture"] as const).map(tab => (
                <button
                  key={tab}
                  onClick={() => setActiveTab(tab)}
                  className={`flex-1 sm:flex-none px-3 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider transition-all ${
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
          <div className="overflow-x-auto -mx-3 sm:mx-0 px-3 sm:px-0">
            <table className="min-w-[720px] w-full text-left border-collapse">
              <thead>
                <tr className="border-b border-slate-100">
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-400">Feature</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-blue-700 bg-blue-50/50 rounded-t-lg">Lumora (Ours)</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-500">Alshmrani et al. (2023)</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-500">Sharma & Guleria (2023)</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-500">Hamamci et al. (2026)</th>
                  <th className="py-3 px-4 text-[10px] uppercase font-mono font-bold text-slate-500">Zhang et al. (2025)</th>
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

                    {/* Hamamci */}
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-1.5">
                        <span className={row.hamamci.check ? "text-emerald-500 text-xs" : "text-rose-400 text-xs"}>
                          {row.hamamci.check ? "✅" : "❌"}
                        </span>
                        <span className="text-xs font-semibold text-slate-600">{row.hamamci.value}</span>
                      </div>
                      {hoveredFeature === row.name && (
                        <div className="text-[10px] text-slate-500 mt-1 max-w-[280px] leading-relaxed">
                          {row.hamamci.detail}
                        </div>
                      )}
                    </td>

                    {/* Zhang */}
                    <td className="py-4 px-4">
                      <div className="flex items-center gap-1.5">
                        <span className={row.zhang.check ? "text-emerald-500 text-xs" : "text-rose-400 text-xs"}>
                          {row.zhang.check ? "✅" : "❌"}
                        </span>
                        <span className="text-xs font-semibold text-slate-600">{row.zhang.value}</span>
                      </div>
                      {hoveredFeature === row.name && (
                        <div className="text-[10px] text-slate-500 mt-1 max-w-[280px] leading-relaxed">
                          {row.zhang.detail}
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
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 sm:gap-6">
          <div className="clinical-card p-3 sm:p-6 lg:col-span-1 bg-slate-900 border-slate-800 text-white animate-fade-in">
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
          <div className="clinical-card p-3 sm:p-6 lg:col-span-2 bg-white flex flex-col justify-between">
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
                <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-3 sm:p-4 space-y-3">
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
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 sm:p-4 flex-1">
                    <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">Alshmrani et al. (VGG19)</div>
                    <div className="text-[11px] font-medium text-slate-700 mt-2 leading-relaxed bg-white border border-slate-100 rounded-lg p-2 shadow-2xs">
                      {scenario.alshmraniOutput}
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 sm:p-4 flex-1">
                    <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">Sharma & Guleria (VGG-16)</div>
                    <div className="text-[11px] font-medium text-slate-700 mt-2 leading-relaxed bg-white border border-slate-100 rounded-lg p-2 shadow-2xs">
                      {scenario.sharmaOutput}
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 sm:p-4 flex-1">
                    <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">Hamamci et al. (CT-CLIP/Chat)</div>
                    <div className="text-[11px] font-medium text-slate-700 mt-2 leading-relaxed bg-white border border-slate-100 rounded-lg p-2 shadow-2xs">
                      {scenario.hamamciOutput}
                    </div>
                  </div>
                  <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 sm:p-4 flex-1">
                    <div className="text-[10px] font-bold text-slate-600 uppercase tracking-wider">Zhang et al. (Grounded VLM)</div>
                    <div className="text-[11px] font-medium text-slate-700 mt-2 leading-relaxed bg-white border border-slate-100 rounded-lg p-2 shadow-2xs">
                      {scenario.zhangOutput}
                    </div>
                  </div>
                </div>
              </div>
            </div>
            
            <div className="text-[10px] text-slate-400 mt-4 pt-3 border-t border-slate-100 flex flex-wrap justify-between items-center gap-2">
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
            <div className="clinical-card p-3 sm:p-5">
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
            <div className="clinical-card p-3 sm:p-5">
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

            {/* Paper 3 details */}
            <div className="clinical-card p-3 sm:p-5">
              <span className="text-[9px] font-mono font-bold bg-slate-100 text-slate-600 px-2 py-0.5 rounded">Reference 03</span>
              <h4 className="text-xs font-bold text-slate-800 mt-2 leading-snug">
                Generalist foundation models from a multimodal dataset for 3D computed tomography
              </h4>
              <p className="text-[10px] font-medium text-slate-400 mt-1 italic">
                Nature Biomedical Engineering (2026)
              </p>

              <div className="mt-4 space-y-2 text-[11px] text-slate-600">
                <p>
                  <strong>Methodology Summary:</strong> CT-CLIP and CT-CHAT foundation models trained on multimodal 3D chest CT volumes. Supports both CT abnormality detection and retrieval tasks alongside language-based interaction.
                </p>
                <div className="bg-slate-50 p-2 rounded-lg border border-slate-100 font-mono text-[10px] text-slate-500">
                  18 CT Abnormality Labels | CT-RATE Dataset | Vision-Language Foundation Models
                </div>
                <p className="text-[10px] text-slate-500">
                  <strong>Key Strengths:</strong> Strong CT modality support with foundation model scale. Includes CT-CHAT for language interaction and 18 anatomically-relevant abnormality labels.
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100">
                <a
                  href="https://doi.org/10.1038/s41551-025-01599-y"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-[10px] font-bold text-blue-600 hover:underline inline-flex items-center gap-1"
                >
                  View Paper via DOI <span>→</span>
                </a>
              </div>
            </div>

            {/* Paper 4 details */}
            <div className="clinical-card p-3 sm:p-5">
              <span className="text-[9px] font-mono font-bold bg-slate-100 text-slate-600 px-2 py-0.5 rounded">Reference 04</span>
              <h4 className="text-xs font-bold text-slate-800 mt-2 leading-snug">
                Development of a large-scale grounded vision language dataset for chest CT analysis
              </h4>
              <p className="text-[10px] font-medium text-slate-400 mt-1 italic">
                Scientific Data, Nature Portfolio (2025)
              </p>

              <div className="mt-4 space-y-2 text-[11px] text-slate-600">
                <p>
                  <strong>Methodology Summary:</strong> RadGenome-Chest CT dataset with grounded VLM baselines. Provides region-level annotations, report supervision, and visual question answering pairs for 3D CT volumes.
                </p>
                <div className="bg-slate-50 p-2 rounded-lg border border-slate-100 font-mono text-[10px] text-slate-500">
                  Region-Grounded Reports | Anatomical Segmentation Labels | VQA Pairs
                </div>
                <p className="text-[10px] text-slate-500">
                  <strong>Key Strengths:</strong> Comprehensive CT dataset with region-level grounding and multiple supervision signals. Supports report generation and visual question answering tasks.
                </p>
              </div>

              <div className="mt-4 pt-3 border-t border-slate-100">
                <a
                  href="https://doi.org/10.1038/s41597-025-05922-9"
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
      <footer className="border-t border-slate-200 bg-white py-5 mt-8 text-center text-[10px] font-medium tracking-wide text-slate-400 px-3">
        <p>
          LUMORA CLINICAL DECISION-SUPPORT SYSTEM — DESIGNED FOR LICENSED HEALTHCARE PROVIDERS
        </p>
      </footer>
    </div>
  );
}
