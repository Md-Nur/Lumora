"use client";

import React, { useState } from "react";

// ─── Model Specifications Data ───────────────────────────────────────────────
const MODEL_SPECS = [
  {
    id: 0,
    name: 'Modality Identification',
    shortName: 'YOLO',
    icon: '🛡️',
    color: 'emerald',
    description: 'Image input validation / guardrail',
    base: 'yolo11s-cls.pt (Ultralytics)',
    params: '~2.8M',
    paramsNum: 2800000,
    activation: 'SiLU',
    dropout: '0.0',
    loss: 'CrossEntropyLoss',
    optimizer: 'Auto (SGD/AdamW)',
    lr: '0.01',
    weightDecay: '0.0005',
    labelSmoothing: '0.0',
    accumSteps: '1',
    batchSize: '128',
    epochs: '100',
    epochsNum: 100,
    inputShape: '224 × 224 (image)',
    trainSize: '2,640 images (80%)',
    trainSizeNum: 2640,
    valSize: '330 images (10%)',
    testSize: '330 images (10%)',
    hfUrl: 'https://huggingface.co/pranto24/xray_ct_scan_identification_model',
    hfName: 'pranto24/xray_ct_scan_identification_model',
    chartUrl: '/yolo_guardrail_metrics.png',
  },
  {
    id: 1,
    name: 'X-Ray VLM',
    shortName: 'X-Ray',
    icon: '🫁',
    color: 'blue',
    description: 'Frontal chest X-ray narrative report generation',
    base: 'DenseNet121 + GPT2',
    params: '~133M',
    paramsNum: 133000000,
    activation: 'GELU (Dec), ReLU (Enc)',
    dropout: '0.1 (Dec), 0.0 (Enc)',
    loss: 'CrossEntropyLoss (shifted)',
    optimizer: 'AdamW',
    lr: 'P1: 1e-4, P2: 2e-5',
    weightDecay: '0.01',
    labelSmoothing: '0.0',
    accumSteps: '1',
    batchSize: '8 / 4 (local)',
    epochs: 'P1:3 + P2:3 = 6',
    epochsNum: 6,
    inputShape: '224×224 img, 128 tok',
    trainSize: '64,592 samples',
    trainSizeNum: 64592,
    valSize: '504 samples (Val)',
    testSize: '338 samples (Test)',
    hfUrl: 'https://huggingface.co/nur9211/mimic-vlm-model',
    hfName: 'nur9211/mimic-vlm-model',
    chartUrl: '/xray_vlm_metrics.png',
  },
  {
    id: 2,
    name: 'CT VLM',
    shortName: 'CT',
    icon: '🧠',
    color: 'violet',
    description: '2D CT slice narrative report generation',
    base: 'DenseNet121 + GPT2',
    params: '~133M',
    paramsNum: 133000000,
    activation: 'GELU (Dec), ReLU (Enc)',
    dropout: '0.1 (Dec), 0.0 (Enc)',
    loss: 'CrossEntropyLoss (shifted)',
    optimizer: 'AdamW',
    lr: 'P1: 1e-4, P2: 2e-5',
    weightDecay: '0.01',
    labelSmoothing: '0.0',
    accumSteps: '1',
    batchSize: '4',
    epochs: 'P1:3 + P2:3 = 6',
    epochsNum: 6,
    inputShape: '224×224 img, 128 tok',
    trainSize: '2,301 samples',
    trainSizeNum: 2301,
    valSize: '106 samples (Val)',
    testSize: '105 samples (Test)',
    hfUrl: 'https://huggingface.co/nur9211/ct-rate-vlm-model',
    hfName: 'nur9211/ct-rate-vlm-model',
    chartUrl: '/ct_vlm_metrics.png',
  },
  {
    id: 3,
    name: 'Translation Model',
    shortName: 'T5',
    icon: '🔤',
    color: 'amber',
    description: 'Clinical jargon → patient-friendly terms',
    base: 't5-small + LoRA',
    params: '~60.8M (LoRA: 295k)',
    paramsNum: 60800000,
    activation: 'ReLU',
    dropout: '0.1',
    loss: 'CrossEntropyLoss (Seq2Seq)',
    optimizer: 'AdamW',
    lr: '3e-4',
    weightDecay: '0.01',
    labelSmoothing: '0.0',
    accumSteps: '4 (eff. batch: 16)',
    batchSize: '4',
    epochs: '5',
    epochsNum: 5,
    inputShape: '512 in / 256 out tok',
    trainSize: '840 samples',
    trainSizeNum: 840,
    valSize: '105 samples (Val)',
    testSize: '105 samples (Test)',
    hfUrl: 'https://huggingface.co/nur9211/lumora_translation',
    hfName: 'nur9211/lumora_translation',
    chartUrl: '/translation_metrics.png',
  },
  {
    id: 4,
    name: 'Disease Detection',
    shortName: 'BERT',
    icon: '🔬',
    color: 'rose',
    description: 'Multi-label 26-disease extraction from reports',
    base: 'Bio_ClinicalBERT',
    params: '~110M',
    paramsNum: 110000000,
    activation: 'GELU',
    dropout: '0.1',
    loss: 'BCEWithLogitsLoss',
    optimizer: 'AdamW',
    lr: '3e-5 (cosine, 10% warm)',
    weightDecay: '0.01',
    labelSmoothing: '0.0',
    accumSteps: '1',
    batchSize: '8',
    epochs: '12',
    epochsNum: 12,
    inputShape: '512 tokens',
    trainSize: '840 samples',
    trainSizeNum: 840,
    valSize: '105 samples (Val)',
    testSize: '105 samples (Test)',
    hfUrl: 'https://huggingface.co/nur9211/lumora_disease_classifier',
    hfName: 'nur9211/lumora_disease_classifier',
    chartUrl: '/disease_detection_metrics.png',
  },
];

// ─── Disease Label Data ──────────────────────────────────────────────────────
const XRAY_DISEASES = [
  'Atelectasis', 'Cardiomegaly', 'Pleural Effusion', 'Pneumonia', 'Pneumothorax',
  'Pulmonary Edema/Vascular Congestion', 'Consolidation', 'No Acute Cardiopulmonary Disease',
  'Pulmonary Fibrosis/Scarring', 'Pulmonary Nodules', 'Rib/Bone Fracture', 'Possible Aspiration',
];

const CT_DISEASES = [
  'Atherosclerosis', 'Emphysema/COPD', 'Hepatic Steatosis', 'Hiatal Hernia',
  'Pericardial Effusion', 'Mosaic Attenuation Pattern', 'Aortic Dilation',
  'Lymphadenopathy', 'Cholelithiasis', 'Osteoporosis', 'Spinal Degenerative Changes',
  'Mild Scoliosis', 'Pulmonary Artery Enlargement', 'Possible Malignancy/Mass',
];

const ALL_LABELS: { label: string; source: 'X-Ray' | 'CT' | 'Both' }[] = [
  { label: 'Aortic Dilation', source: 'CT' },
  { label: 'Atelectasis', source: 'Both' },
  { label: 'Atherosclerosis', source: 'CT' },
  { label: 'Cardiomegaly', source: 'Both' },
  { label: 'Cholelithiasis', source: 'CT' },
  { label: 'Consolidation', source: 'Both' },
  { label: 'Emphysema/COPD', source: 'CT' },
  { label: 'Hepatic Steatosis', source: 'CT' },
  { label: 'Hiatal Hernia', source: 'CT' },
  { label: 'Lymphadenopathy', source: 'CT' },
  { label: 'Mild Scoliosis', source: 'CT' },
  { label: 'Mosaic Attenuation Pattern', source: 'CT' },
  { label: 'No Acute Cardiopulmonary Disease', source: 'Both' },
  { label: 'Osteoporosis', source: 'CT' },
  { label: 'Pericardial Effusion', source: 'CT' },
  { label: 'Pleural Effusion', source: 'Both' },
  { label: 'Pneumonia', source: 'Both' },
  { label: 'Pneumothorax', source: 'X-Ray' },
  { label: 'Possible Aspiration', source: 'Both' },
  { label: 'Possible Malignancy/Mass', source: 'CT' },
  { label: 'Pulmonary Artery Enlargement', source: 'CT' },
  { label: 'Pulmonary Edema/Vascular Congestion', source: 'Both' },
  { label: 'Pulmonary Fibrosis/Scarring', source: 'Both' },
  { label: 'Pulmonary Nodules', source: 'Both' },
  { label: 'Rib/Bone Fracture', source: 'Both' },
  { label: 'Spinal Degenerative Changes', source: 'CT' },
];

const colorMap: Record<string, { bg: string; border: string; text: string; barBg: string; lightBg: string }> = {
  emerald: { bg: 'bg-emerald-50', border: 'border-emerald-200', text: 'text-emerald-700', barBg: 'bg-emerald-500', lightBg: 'bg-emerald-100' },
  blue: { bg: 'bg-blue-50', border: 'border-blue-200', text: 'text-blue-700', barBg: 'bg-blue-500', lightBg: 'bg-blue-100' },
  violet: { bg: 'bg-violet-50', border: 'border-violet-200', text: 'text-violet-700', barBg: 'bg-violet-500', lightBg: 'bg-violet-100' },
  amber: { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', barBg: 'bg-amber-500', lightBg: 'bg-amber-100' },
  rose: { bg: 'bg-rose-50', border: 'border-rose-200', text: 'text-rose-700', barBg: 'bg-rose-500', lightBg: 'bg-rose-100' },
};

// ─── Detail Row Helper ───────────────────────────────────────────────────────
function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col min-[360px]:flex-row min-[360px]:justify-between gap-0.5 min-[360px]:gap-3 items-start min-[360px]:items-baseline py-1.5 border-b border-slate-50 last:border-0">
      <span className="text-[11px] text-slate-500 font-medium shrink-0">{label}</span>
      <span className="text-[11px] font-mono text-slate-800 font-semibold min-[360px]:text-right break-words min-[360px]:max-w-[60%]">{value}</span>
    </div>
  );
}

// ─── Comparison Bar Chart ────────────────────────────────────────────────────
function ComparisonChart({ title, dataKey, formatter }: { title: string; dataKey: 'paramsNum' | 'trainSizeNum' | 'epochsNum'; formatter: (v: number) => string }) {
  const maxVal = Math.max(...MODEL_SPECS.map(m => m[dataKey]));
  return (
    <div className="clinical-card p-3 sm:p-5">
      <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4">{title}</h3>
      <div className="space-y-3">
        {MODEL_SPECS.map(m => {
          const pct = maxVal > 0 ? (m[dataKey] / maxVal) * 100 : 0;
          const c = colorMap[m.color];
          return (
            <div key={m.id} className="grid grid-cols-[3.5rem_minmax(0,1fr)] min-[420px]:grid-cols-[4rem_minmax(0,1fr)_5.5rem] items-center gap-2 min-[420px]:gap-3">
              <span className="text-[11px] font-semibold text-slate-600 truncate">{m.shortName}</span>
              <div className="flex-1 h-6 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full ${c.barBg} transition-all duration-700 ease-out`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="col-start-2 min-[420px]:col-start-auto text-[11px] font-mono text-slate-700 font-semibold min-[420px]:text-right shrink-0">{formatter(m[dataKey])}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ─── Disease Panel ───────────────────────────────────────────────────────────
function DiseaseLabelPanel() {
  const [tab, setTab] = useState<'all' | 'xray' | 'ct'>('all');

  const tabs = [
    { key: 'all', label: '🔬 All 26 Labels', count: 26 },
    { key: 'xray', label: '🩻 X-Ray Only', count: XRAY_DISEASES.length },
    { key: 'ct', label: '🧠 CT Scan Focused', count: CT_DISEASES.length },
  ] as const;

  const displayed = tab === 'all' ? ALL_LABELS
    : tab === 'xray' ? ALL_LABELS.filter(d => d.source === 'X-Ray' || d.source === 'Both')
    : ALL_LABELS.filter(d => d.source === 'CT' || d.source === 'Both');

  const sourceBadge = (source: 'X-Ray' | 'CT' | 'Both') => {
    if (source === 'X-Ray') return <span className="ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-blue-100 text-blue-700 border border-blue-200 shrink-0">X-Ray</span>;
    if (source === 'CT') return <span className="ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-violet-100 text-violet-700 border border-violet-200 shrink-0">CT</span>;
    return <span className="ml-auto text-[9px] font-bold px-1.5 py-0.5 rounded-full bg-emerald-100 text-emerald-700 border border-emerald-200 shrink-0">Both</span>;
  };

  return (
    <div className="md:col-span-2 clinical-card p-3 sm:p-6">
      <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4 flex flex-wrap items-center gap-2">
        <span>🏷️ Pathology Label Vocabulary</span>
        <span className="sm:ml-auto text-[10px] font-normal text-slate-400 normal-case">Trained on MIMIC-CXR + CT-RATE</span>
      </h3>

      {/* Dataset info strip */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-4">
        <div className="rounded-xl border border-blue-100 bg-blue-50 p-3">
          <div className="text-[10px] font-bold text-blue-700 uppercase tracking-wider mb-1">🩻 MIMIC-CXR (X-Ray)</div>
          <div className="text-[11px] text-slate-600">Frontal chest radiograph reports. Captures cardiac, pulmonary, and pleural findings visible in 2D X-rays.</div>
        </div>
        <div className="rounded-xl border border-violet-100 bg-violet-50 p-3">
          <div className="text-[10px] font-bold text-violet-700 uppercase tracking-wider mb-1">🧠 CT-RATE (CT Scan)</div>
          <div className="text-[11px] text-slate-600">Thoracic CT reports. Includes deeper anatomical structures — abdominal organs, vascular, musculoskeletal findings.</div>
        </div>
      </div>

      {/* Tab selector */}
      <div className="flex flex-col min-[420px]:flex-row gap-1 bg-slate-100 rounded-lg p-1 mb-4 w-full">
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setTab(t.key)}
            className={`flex-1 py-1.5 px-2 rounded-md text-[10px] font-bold transition-all duration-150 flex items-center justify-center gap-1 ${
              tab === t.key ? 'bg-white text-slate-800 shadow-xs border border-slate-200' : 'text-slate-500 hover:text-slate-700'
            }`}
          >
            {t.label}
            <span className="bg-slate-200 text-slate-600 rounded-full px-1.5 py-0.5 text-[9px] font-mono">{t.count}</span>
          </button>
        ))}
      </div>

      {/* Disease pill grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5">
        {displayed.map((d, i) => (
          <div key={d.label} className="flex items-center gap-2 bg-slate-50 border border-slate-100 rounded-lg px-3 py-2 hover:border-slate-200 hover:bg-white transition-all duration-150">
            <span className="text-[10px] font-mono text-slate-400 w-5 shrink-0">{String(i + 1).padStart(2, '0')}</span>
            <span className="text-[11px] font-semibold text-slate-700 leading-tight">{d.label}</span>
            {sourceBadge(d.source)}
          </div>
        ))}
      </div>

      <div className="mt-4 pt-3 border-t border-slate-100 flex flex-wrap items-center gap-3 sm:gap-4 text-[10px] text-slate-400">
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-emerald-200 border border-emerald-300 inline-block"></span> Both datasets</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-blue-200 border border-blue-300 inline-block"></span> MIMIC-CXR only</span>
        <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-full bg-violet-200 border border-violet-300 inline-block"></span> CT-RATE focused</span>
        <span className="sm:ml-auto">Threshold: 0.15 · Multi-label sigmoid · Post-processing: mutual-exclusion logic</span>
      </div>
    </div>
  );
}

export default function ModelSpecifications() {
  const [selectedModel, setSelectedModel] = useState<number>(0);
  const model = MODEL_SPECS[selectedModel];
  const c = colorMap[model.color];

  return (
    <div className="flex flex-col bg-[#f8fafc] font-sans antialiased text-slate-800">

      <main className="flex-1 max-w-7xl w-full mx-auto px-3 sm:px-6 py-4 sm:py-6 space-y-4 sm:space-y-6">
        {/* Model Selector Cards */}
        <div className="grid grid-cols-1 min-[340px]:grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          {MODEL_SPECS.map(m => {
            const mc = colorMap[m.color];
            const isActive = m.id === selectedModel;
            return (
              <button
                key={m.id}
                onClick={() => setSelectedModel(m.id)}
                className={`group relative p-3 sm:p-4 rounded-xl border-2 transition-all duration-200 text-left ${
                  isActive
                    ? `${mc.bg} ${mc.border} shadow-md scale-[1.02]`
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:shadow-md hover:scale-[1.02]'
                }`}
              >
                <div className="text-2xl mb-2">{m.icon}</div>
                <div className={`text-xs font-bold ${isActive ? mc.text : 'text-slate-700'}`}>{m.name}</div>
                <div className="text-[10px] font-mono text-slate-400 mt-1">{m.params}</div>
                {isActive && (
                  <div className={`absolute top-2 right-2 h-2 w-2 rounded-full ${mc.barBg}`} />
                )}
              </button>
            );
          })}
        </div>

        {/* Detail Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Overview */}
          <div className={`clinical-card p-3 sm:p-5 border-t-4 ${c.border}`}>
            <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4 flex items-center gap-2">
              <span className="text-base">{model.icon}</span> Overview
            </h3>
            <DetailRow label="Model Name" value={model.name} />
            <DetailRow label="Description" value={model.description} />
            <DetailRow label="Base Architecture" value={model.base} />
            <DetailRow label="Parameters" value={model.params} />
            <div className="mt-3 pt-3 border-t border-slate-100">
              <a
                href={model.hfUrl}
                target="_blank"
                rel="noopener noreferrer"
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-[11px] font-bold ${c.lightBg} ${c.text} border ${c.border} hover:shadow-md transition-all`}
              >
                <svg className="w-3.5 h-3.5" viewBox="0 0 120 120" fill="currentColor"><path d="M37.6 60.7c0-4.7-3.8-8.5-8.5-8.5s-8.5 3.8-8.5 8.5v23.8c0 4.7 3.8 8.5 8.5 8.5s8.5-3.8 8.5-8.5V60.7zm62.8 0c0-4.7-3.8-8.5-8.5-8.5s-8.5 3.8-8.5 8.5v23.8c0 4.7 3.8 8.5 8.5 8.5s8.5-3.8 8.5-8.5V60.7z"/><path d="M96.5 43.8c-2.3-6.5-6.3-12.2-11.5-16.7l7.8-7.8c1.7-1.7 1.7-4.4 0-6.1s-4.4-1.7-6.1 0l-8.8 8.8C72.3 18.1 66.3 16 60 16s-12.3 2.1-17.9 6l-8.8-8.8c-1.7-1.7-4.4-1.7-6.1 0s-1.7 4.4 0 6.1l7.8 7.8c-5.2 4.5-9.2 10.2-11.5 16.7-1.6 4.5-2.5 9.4-2.5 14.5v2.2c0 3.3 2.7 6 6 6h66c3.3 0 6-2.7 6-6v-2.2c0-5.1-.9-10-2.5-14.5zM44.3 46.7c-2.3 0-4.3-1.9-4.3-4.3s1.9-4.3 4.3-4.3 4.3 1.9 4.3 4.3-2 4.3-4.3 4.3zm31.4 0c-2.3 0-4.3-1.9-4.3-4.3s1.9-4.3 4.3-4.3 4.3 1.9 4.3 4.3-2 4.3-4.3 4.3z"/></svg>
                {model.hfName}
              </a>
            </div>
          </div>

          {/* Training Configuration */}
          <div className="clinical-card p-3 sm:p-5">
            <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4">
              ⚙️ Training Configuration
            </h3>
            <DetailRow label="Epochs" value={model.epochs} />
            <DetailRow label="Batch Size" value={model.batchSize} />
            <DetailRow label="Learning Rate" value={model.lr} />
            <DetailRow label="Optimizer" value={model.optimizer} />
            <DetailRow label="Weight Decay" value={model.weightDecay} />
            <DetailRow label="Gradient Accum Steps" value={model.accumSteps} />
            <DetailRow label="Label Smoothing" value={model.labelSmoothing} />
          </div>

          {/* Architecture */}
          <div className="clinical-card p-3 sm:p-5">
            <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4">
              🏗️ Architecture
            </h3>
            <DetailRow label="Activation" value={model.activation} />
            <DetailRow label="Dropout" value={model.dropout} />
            <DetailRow label="Loss Function" value={model.loss} />
            <DetailRow label="Input Shape" value={model.inputShape} />
          </div>

          {/* Dataset & Splits */}
          <div className="clinical-card p-3 sm:p-5">
            <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4">
              📊 Dataset & Splits
            </h3>
            <DetailRow label="Training Set Size" value={model.trainSize} />
            <DetailRow label="Validation Set Size" value={model.valSize} />
            <DetailRow label="Test Set Size" value={model.testSize} />
          </div>

          {/* Model Performance Analytics */}
          {model.chartUrl && (
            <div className="md:col-span-2 clinical-card p-3 sm:p-6 flex flex-col items-center">
              <h3 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4 w-full flex items-center gap-2">
                📊 Model Performance Analytics
              </h3>
              <div className="w-full max-w-2xl bg-slate-50 border border-slate-100 rounded-xl p-3 shadow-2xs hover:shadow-sm transition-all duration-300">
                <img
                  src={model.chartUrl}
                  alt={`${model.name} Metrics Chart`}
                  className="w-full h-auto rounded-lg object-contain mix-blend-multiply"
                />
              </div>
              <p className="text-[11px] text-slate-500 mt-3 font-medium max-w-prose text-center leading-relaxed">
                {model.id === 0 && "Confusion matrix evaluating the modality classifier on test split. High specificity prevents non-medical scans from invoking downstream VLM checkpoints."}
                {model.id === 1 && "Cross-entropy loss curves across Phase 1 (frozen visual encoder, projector tuning) and Phase 2 (joint end-to-end VLM fine-tuning) on the MIMIC-CXR dataset."}
                {model.id === 2 && "Comparison of validation loss and perplexity (PPL) on conversion of NIfTI CT volumes to representative axial 2D slices for report generation."}
                {model.id === 3 && "T5 Transformer sequence-to-sequence training and validation loss curves (dual y-axes) indicating stable parameter-efficient fine-tuning with LoRA."}
                {model.id === 4 && "Micro-averaged precision, recall, and F1 scores of the Bio_ClinicalBERT pathology classifier. High recall configuration ensures safety by capturing potential findings."}
              </p>
            </div>
          )}
        </div>

        {/* Disease Label Panel — only for ClinicalBERT */}
        {model.id === 4 && (
          <div className="grid grid-cols-1">
            <DiseaseLabelPanel />
          </div>
        )}

        {/* Comparison Analytics */}
        <div>
          <h2 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono mb-4">
            📈 Cross-Model Comparison
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <ComparisonChart
              title="Parameter Count"
              dataKey="paramsNum"
              formatter={(v) => v >= 1_000_000 ? `${(v / 1_000_000).toFixed(1)}M` : `${(v / 1_000).toFixed(0)}K`}
            />
            <ComparisonChart
              title="Training Dataset Size"
              dataKey="trainSizeNum"
              formatter={(v) => v >= 1_000 ? `${(v / 1_000).toFixed(1)}K` : `${v}`}
            />
            <ComparisonChart
              title="Training Epochs"
              dataKey="epochsNum"
              formatter={(v) => `${v}`}
            />
          </div>
        </div>
      </main>

    </div>
  );
}
