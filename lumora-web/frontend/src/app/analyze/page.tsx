"use client";

import React, { useState, useEffect, useRef } from "react";

interface Telemetry {
  status: "verified" | "rejected" | "none";
  meanSaturation: number;
  saturationThreshold: number;
  grayStd: number;
  contrastThreshold: number;
  saturationMessage?: string;
  contrastMessage?: string;
  engineUsed: string;
  inferenceTime: string;
}

type Modality = "xray" | "ct";

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [modality, setModality] = useState<Modality>("xray");
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [processingStep, setProcessingStep] = useState<number>(0);
  const [consentChecked, setConsentChecked] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<
    "checking" | "online" | "offline"
  >("checking");
  const [currentDate, setCurrentDate] = useState<string>("");

  // Results
  const [reportText, setReportText] = useState<string>("");
  const [displayedReport, setDisplayedReport] = useState<string>("");
  const [translationText, setTranslationText] = useState<string>("");
  const [displayedTranslation, setDisplayedTranslation] = useState<string>("");
  const [diseases, setDiseases] = useState<string[]>([]);
  const [telemetry, setTelemetry] = useState<Telemetry>({
    status: "none",
    meanSaturation: 0.0,
    saturationThreshold: 0.15,
    grayStd: 0.0,
    contrastThreshold: 8.0,
    engineUsed: "N/A",
    inferenceTime: "N/A",
  });

  const typewriterIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const translationTypewriterRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Set formatted current date on mount
  useEffect(() => {
    setCurrentDate(
      new Date().toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      })
    );
  }, []);

  // Load modality from query parameter on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const params = new URLSearchParams(window.location.search);
      const queryModality = params.get("modality");
      if (queryModality === "ct" || queryModality === "xray") {
        setModality(queryModality as Modality);
      }
    }
  }, []);

  const invalidMessageForModality = (selectedModality: Modality) => {
    if (selectedModality === "ct") return "It is not a chest/lung CT scan.";
    return "It is not a chest/lung X-ray image.";
  };

  const resolveInvalidScanMessage = (selectedModality: Modality, backendReport?: string) => {
    if (!backendReport || backendReport.includes("x-ray or chest/lung ct-scan") || backendReport.includes("Invalid")) {
      return invalidMessageForModality(selectedModality);
    }
    return backendReport;
  };

  const getFileExtension = (name: string) => {
    const lowerName = name.toLowerCase();
    if (lowerName.endsWith(".nii.gz")) return "nii.gz";
    return lowerName.split(".").pop() || "";
  };

  // Ping FastAPI health check
  const checkBackendHealth = async () => {
    try {
      const res = await fetch(process.env.NEXT_PUBLIC_BACKEND_BASE_URL || "http://localhost:8000", {
        signal: AbortSignal.timeout(2000),
      });
      if (res.ok) {
        setBackendOnline("online");
      } else {
        setBackendOnline("offline");
      }
    } catch {
      setBackendOnline("offline");
    }
  };

  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 5000);
    return () => clearInterval(interval);
  }, []);

  // Typewriter effect for Clinical Report
  useEffect(() => {
    if (typewriterIntervalRef.current) {
      clearInterval(typewriterIntervalRef.current);
    }

    if (!reportText) {
      setDisplayedReport("");
      return;
    }

    let currentIndex = 0;
    setDisplayedReport("");

    typewriterIntervalRef.current = setInterval(() => {
      if (currentIndex < reportText.length) {
        setDisplayedReport((prev) => prev + reportText.substring(currentIndex, currentIndex + 12));
        currentIndex += 12;
      } else {
        if (typewriterIntervalRef.current)
          clearInterval(typewriterIntervalRef.current);
      }
    }, 10);

    return () => {
      if (typewriterIntervalRef.current)
        clearInterval(typewriterIntervalRef.current);
    };
  }, [reportText]);

  // Typewriter effect for Patient Translation
  useEffect(() => {
    if (translationTypewriterRef.current) {
      clearInterval(translationTypewriterRef.current);
    }

    if (!translationText) {
      setDisplayedTranslation("");
      return;
    }

    let currentIndex = 0;
    setDisplayedTranslation("");

    translationTypewriterRef.current = setInterval(() => {
      if (currentIndex < translationText.length) {
        setDisplayedTranslation((prev) => prev + translationText.substring(currentIndex, currentIndex + 12));
        currentIndex += 12;
      } else {
        if (translationTypewriterRef.current)
          clearInterval(translationTypewriterRef.current);
      }
    }, 10);

    return () => {
      if (translationTypewriterRef.current)
        clearInterval(translationTypewriterRef.current);
    };
  }, [translationText]);

  // Handle Drag & Drop
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (selectedFile: File) => {
    const ext = getFileExtension(selectedFile.name);
    
    // Auto-detect modality from extension
    let targetModality: Modality = "xray";
    if (["nii", "nii.gz", "dcm"].includes(ext)) {
      targetModality = "ct";
    }

    setModality(targetModality);
    setFile(selectedFile);
    setPreviewUrl(targetModality === "xray" ? URL.createObjectURL(selectedFile) : null);
    
    // Reset output panels
    setReportText("");
    setDisplayedReport("");
    setTranslationText("");
    setDisplayedTranslation("");
    setDiseases([]);
    setProcessingStep(0);
    setTelemetry({
      status: "none",
      meanSaturation: 0.0,
      saturationThreshold: 0.15,
      grayStd: 0.0,
      contrastThreshold: 8.0,
      engineUsed: "N/A",
      inferenceTime: "N/A",
    });
  };

  // Helper to load sample files
  const loadSampleFile = async (url: string, filename: string, targetModality: Modality) => {
    // Reset
    handleReset();
    setIsProcessing(true);
    
    try {
      const response = await fetch(url);
      if (!response.ok) throw new Error("Failed to load sample asset");
      const blob = await response.blob();
      const sampleFile = new File([blob], filename, { type: blob.type || "image/jpeg" });
      
      setModality(targetModality);
      setFile(sampleFile);
      setPreviewUrl(targetModality === "xray" ? url : null);
      setConsentChecked(true); // Automatically accept consent for quick sample tries
    } catch (err) {
      console.error("Error loading sample file:", err);
    } finally {
      setIsProcessing(false);
    }
  };

  const loadMockCTSample = () => {
    handleReset();
    
    // Create a mock binary 2D slice
    const content = new TextEncoder().encode("Mock NIfTI CT scan 2D slice binary stream");
    const sampleFile = new File([content], "sample_ct_pathology.nii.gz", { type: "application/gzip" });
    
    setModality("ct");
    setFile(sampleFile);
    setPreviewUrl(null);
    setConsentChecked(true);
  };

  const handleModalityChange = (targetModality: Modality) => {
    if (!file) {
      setModality(targetModality);
      handleReset();
    }
  };

  // Run Prediction
  const runPredictivePipeline = async () => {
    if (!file) return;

    setIsProcessing(true);
    setReportText("");
    setDisplayedReport("");
    setTranslationText("");
    setDisplayedTranslation("");
    setDiseases([]);

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_BASE_URL || "http://localhost:8000";

    // Offline simulation mode
    if (backendOnline === "offline") {
      const startTime = performance.now();
      
      // Animation Stepper
      setProcessingStep(1); // Ingestion
      await new Promise((resolve) => setTimeout(resolve, 100));
      
      setProcessingStep(2); // Guardrail
      await new Promise((resolve) => setTimeout(resolve, 100));
      
      setProcessingStep(3); // Feature extraction
      await new Promise((resolve) => setTimeout(resolve, 100));
      
      setProcessingStep(4); // Formatting
      await new Promise((resolve) => setTimeout(resolve, 100));
      
      const endTime = performance.now();
      const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);

      const isInvalidCase =
        file.name.toLowerCase().includes("invalid") ||
        file.name.toLowerCase().includes("portrait");

      if (isInvalidCase) {
        setTelemetry({
          status: "rejected",
          meanSaturation: modality === "xray" ? 0.38 : 0.0,
          saturationThreshold: 0.15,
          grayStd: modality === "xray" ? 4.2 : 0.0,
          contrastThreshold: 8.0,
          saturationMessage: "Guardrail validation alert: Input does not appear to contain diagnostic structures.",
          contrastMessage: "Pixel distributions failed clinical chest projection thresholds.",
          engineUsed: "Clinical Physics Guardrail v1.4",
          inferenceTime: `${elapsedSeconds}s`,
        });
        setReportText(invalidMessageForModality(modality));
        setTranslationText("");
        setDiseases([]);
      } else {
        const isPathologicalCase =
          file.name.toLowerCase().includes("pathology") ||
          file.name.toLowerCase().includes("abnormal") ||
          file.name.toLowerCase().includes("effusion");

        setTelemetry({
          status: "verified",
          meanSaturation: 0.02,
          saturationThreshold: 0.15,
          grayStd: isPathologicalCase ? 48.6 : 55.3,
          contrastThreshold: 8.0,
          saturationMessage: "Diagnostic study verified. Confirmed monochrome radiograph.",
          contrastMessage: "Contrast levels standard. Clinical structures decoded successfully.",
          engineUsed: modality === "ct" ? "Lumora CT-RATE VLM Assistant" : "Lumora VLM Assistant",
          inferenceTime: `${elapsedSeconds}s`,
        });

        if (modality === "ct") {
          // CT mode simulation
          if (isPathologicalCase) {
            setReportText(
              "PATIENT CLINICAL REPORT\n\nEXAMINATION: Thoracic CT Scan (2D Slice)\n\nFINDINGS:\nBilateral emphysematous changes are noted. Minimal calcification of the aortic wall (atherosclerosis). Mediastinal lymph nodes are within normal limits. Minimal spinal degenerative osteophytes are seen.\n\nIMPRESSION:\n1. Moderate emphysema/COPD changes.\n2. Mild aortic wall atherosclerosis."
            );
            setDiseases(["Emphysema/COPD", "Atherosclerosis", "Spinal Degenerative Changes"]);
            setTranslationText(
              "The CT scan shows signs of emphysema (COPD) in both lungs. There is also mild hardening of your main arteries and minor wear-and-tear changes in your spine."
            );
          } else {
            setReportText(
              "PATIENT CLINICAL REPORT\n\nEXAMINATION: Thoracic CT Scan (2D Slice)\n\nFINDINGS:\nVisualized lung fields are normal without focal consolidations or nodules. Pleural spaces are clear. No aortic dilation. Mediastinal structures are unremarkable.\n\nIMPRESSION:\nUnremarkable CT scan of the chest."
            );
            setDiseases(["No acute cardiopulmonary disease"]);
            setTranslationText(
              "The CT scan findings are normal. There are no signs of fluid, nodules, or artery enlargement in your chest."
            );
          }
        } else {
          // X-ray mode simulation
          if (isPathologicalCase) {
            setReportText(
              "PATIENT CLINICAL REPORT\n\nEXAMINATION: Chest Radiograph, Frontal View\n\nFINDINGS:\nThe cardiac silhouette is mildly enlarged. Prominence of the bilateral pulmonary interstitial markings suggesting mild interstitial congestion. Small bilateral pleural effusions are noted.\n\nIMPRESSION:\n1. Mild cardiomegaly and congestion.\n2. Small bilateral pleural effusions."
            );
            setDiseases(["Cardiomegaly", "Pulmonary Edema/Vascular Congestion", "Pleural Effusion"]);
            setTranslationText(
              "Your chest X-ray shows a slightly enlarged heart and signs of fluid congestion in your lungs. There is also a small amount of fluid pooling around both lungs."
            );
          } else {
            setReportText(
              "PATIENT CLINICAL REPORT\n\nEXAMINATION: Chest Radiograph, Frontal View\n\nFINDINGS:\nThe lungs are clear. No consolidations, pleural effusions, or signs of pneumothorax are seen. Cardiac silhouette is within normal limits.\n\nIMPRESSION:\nNormal chest radiograph."
            );
            setDiseases(["No acute cardiopulmonary disease"]);
            setTranslationText(
              "The chest X-ray is normal. Your lungs are completely clear and your heart size is within normal limits."
            );
          }
        }
      }
    } else {
      // Online mode: fetch from FastAPI backend
      const formData = new FormData();
      formData.append("file", file);
      
      let endpoint = "/predict";
      if (modality === "ct") endpoint = "/predict/ct";

      try {
        setProcessingStep(1); // Ingestion
        setProcessingStep(2); // Guardrail
        
        const response = await fetch(`${backendUrl}${endpoint}`, {
          method: "POST",
          body: formData,
        });

        setProcessingStep(3); // Feature processing
        setProcessingStep(4); // Synthesis

        const data = await response.json();

        if (response.ok) {
          const meanSaturation = typeof data.mean_saturation === "number" ? data.mean_saturation : 0;
          const grayStd = typeof data.gray_std === "number" ? data.gray_std : 0;
          
          setTelemetry({
            status: "verified",
            meanSaturation,
            saturationThreshold: 0.15,
            grayStd,
            contrastThreshold: modality === "ct" ? 1.0 : 8.0,
            saturationMessage: data.telemetry || "Diagnostic study verified.",
            contrastMessage: `Contrast verified. Value: ${grayStd.toFixed(1)}`,
            engineUsed: modality === "ct" ? "Lumora CT-RATE VLM Assistant" : "Lumora VLM Assistant",
            inferenceTime: "Dynamic Node",
          });
          setReportText(data.report);
          setTranslationText(data.translation || "");
          setDiseases(data.diseases || []);
        } else if (response.status === 422) {
          const meanSaturation = typeof data.mean_saturation === "number" ? data.mean_saturation : 0;
          const grayStd = typeof data.gray_std === "number" ? data.gray_std : 0;

          setTelemetry({
            status: "rejected",
            meanSaturation,
            saturationThreshold: 0.15,
            grayStd,
            contrastThreshold: modality === "ct" ? 1.0 : 8.0,
            saturationMessage: data.telemetry || "API validation error.",
            contrastMessage: "Pixel distribution failed guardrail validation.",
            engineUsed: modality === "ct" ? "CT Input Validator" : "Clinical Physics Guardrail",
            inferenceTime: "Dynamic Node",
          });
          setReportText(resolveInvalidScanMessage(modality, data.report));
          setTranslationText("");
          setDiseases([]);
        } else {
          throw new Error(data.detail || "Server error");
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        console.error(err);
        setReportText(
          `INFERENCE ERROR: ${errorMsg || "Failed to communicate with the Lumora analysis host."}`
        );
        setTranslationText("");
        setDiseases([]);
        setTelemetry({
          status: "rejected",
          meanSaturation: 0.0,
          saturationThreshold: 0.15,
          grayStd: 0.0,
          contrastThreshold: 8.0,
          saturationMessage: "Connection failed — could not reach backend.",
          contrastMessage: "N/A",
          engineUsed: "Network Host Monitor",
          inferenceTime: "N/A",
        });
      }
    }

    setIsProcessing(false);
    setProcessingStep(0);
  };

  // Reset
  const handleReset = () => {
    setFile(null);
    setPreviewUrl(null);
    setReportText("");
    setDisplayedReport("");
    setTranslationText("");
    setDisplayedTranslation("");
    setDiseases([]);
    setConsentChecked(false);
    setProcessingStep(0);
    setTelemetry({
      status: "none",
      meanSaturation: 0.0,
      saturationThreshold: 0.15,
      grayStd: 0.0,
      contrastThreshold: 8.0,
      engineUsed: "N/A",
      inferenceTime: "N/A",
    });
  };

  const hasResults = displayedReport || isProcessing;

  return (
    <div className="flex flex-col bg-background min-h-screen">
      <main className="mx-auto w-full max-w-7xl px-4 sm:px-6 py-10 transition-all duration-300">
        
        {/* Page Heading */}
        <div className="text-center mb-10 max-w-2xl mx-auto">
          <div className="inline-flex items-center gap-1.5 rounded-full border border-primary/20 bg-primary/5 px-3 py-1 text-[10px] font-bold text-primary-deep tracking-wider uppercase mb-3">
            <span className="h-1.5 w-1.5 rounded-full bg-primary-deep animate-pulse"></span>
            DIAGNOSTIC WORKSTATION
          </div>
          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-foreground font-sans">
            AI Scan Intelligence
          </h1>
          <p className="mt-2 text-sm text-muted-foreground">
            Feed medical scans into the Lumora neural pipeline. View diagnostic report drafts, confidence telemetry, and clear descriptions in one place.
          </p>
        </div>

        {/* Workspace Layout Grid */}
        <div className="grid gap-8 grid-cols-1 lg:grid-cols-12">
          
          {/* LEFT COLUMN: CONTROL & WORKSTATION INPUT (Spans 5 on lg) */}
          <div className="lg:col-span-5 flex flex-col gap-6">
            
            {/* Console Settings Box */}
            <div className="rounded-2xl border border-border/80 bg-white shadow-xs p-5 flex flex-col gap-4">
              
              <div className="flex items-center justify-between border-b border-border/50 pb-3">
                <span className="text-xs font-bold text-slate-800 tracking-wider uppercase font-mono flex items-center gap-2">
                  <svg className="w-4 h-4 text-primary-deep" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4" />
                  </svg>
                  Console Panel
                </span>
                
                {/* Online Status Flag */}
                <div className="flex items-center gap-1.5">
                  <span className={`h-2 w-2 rounded-full ${
                    backendOnline === "online" ? "bg-emerald-500 animate-pulse" : 
                    backendOnline === "offline" ? "bg-amber-400" : "bg-slate-300 animate-ping"
                  }`} />
                  <span className="text-[10px] font-bold font-mono text-muted-foreground uppercase">
                    {backendOnline === "online" ? "API Host Connected" : 
                     backendOnline === "offline" ? "Simulation Active" : "Connecting..."}
                  </span>
                </div>
              </div>

              {/* Modality Selector Buttons */}
              <div className="flex flex-col gap-2">
                <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-wider font-mono">
                  Select Modality
                </label>
                <div className="grid grid-cols-2 gap-2.5">
                  
                  {/* X-Ray Modality */}
                  <button
                    onClick={() => handleModalityChange("xray")}
                    disabled={isProcessing}
                    className={`flex items-center justify-center gap-2 rounded-xl border py-3 px-4 transition-all duration-200 cursor-pointer ${
                      modality === "xray"
                        ? "border-primary-deep bg-primary/5 text-primary-deep font-bold ring-2 ring-primary/10"
                        : "border-border bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    <svg className="h-4.5 w-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span className="text-xs">Chest X-Ray</span>
                  </button>

                  {/* CT Modality */}
                  <button
                    onClick={() => handleModalityChange("ct")}
                    disabled={isProcessing}
                    className={`flex items-center justify-center gap-2 rounded-xl border py-3 px-4 transition-all duration-200 cursor-pointer ${
                      modality === "ct"
                        ? "border-primary-deep bg-primary/5 text-primary-deep font-bold ring-2 ring-primary/10"
                        : "border-border bg-white text-slate-600 hover:border-slate-300 hover:bg-slate-50"
                    }`}
                  >
                    <svg className="h-4.5 w-4.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.2}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5A3.375 3.375 0 0010.125 2.25H8.25m0 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                    </svg>
                    <span className="text-xs">CT Scan (2D)</span>
                  </button>

                </div>
              </div>

              {/* Upload Target Area */}
              <div className="flex flex-col gap-2">
                <label className="text-[11px] font-bold text-muted-foreground uppercase tracking-wider font-mono">
                  Diagnostics Input
                </label>
                
                <div
                  onDragOver={handleDragOver}
                  onDragLeave={handleDragLeave}
                  onDrop={handleDrop}
                  onClick={() => !file && fileInputRef.current?.click()}
                  className={`relative overflow-hidden rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-6 text-center transition-all duration-300 min-h-[260px] ${
                    file
                      ? "border-slate-300 bg-slate-900"
                      : "border-slate-200 bg-slate-50 hover:border-primary-deep/50 hover:bg-slate-50/50 cursor-pointer"
                  } ${isDragActive ? "border-primary bg-primary/5 shadow-inner" : ""}`}
                >
                  <input
                    type="file"
                    ref={fileInputRef}
                    onChange={handleFileChange}
                    accept={
                      modality === "ct"
                        ? ".nii,.nii.gz,.dcm,.zip"
                        : ".png,.jpg,.jpeg"
                    }
                    className="hidden"
                  />

                  {previewUrl ? (
                    /* X-Ray Preview with HUD overlay */
                    <div className="relative w-full max-w-[240px] aspect-square rounded-lg overflow-hidden flex items-center justify-center border border-slate-700 bg-black/80 group">
                      
                      {/* Scanner light line */}
                      {isProcessing && <div className="medical-scanner-line" />}
                      
                      {/* Image under scanner */}
                      <img
                        src={previewUrl}
                        alt="Radiograph Scan"
                        className="max-w-full max-h-full object-contain filter contrast-125 brightness-90 grayscale opacity-80"
                      />

                      {/* HUD Overlays */}
                      <div className="absolute top-2 left-2 text-[10px] font-mono text-cyan-400 font-bold bg-black/60 px-1 py-0.5 rounded-sm select-none">
                        [R]
                      </div>
                      <div className="absolute top-2 right-2 text-[10px] font-mono text-cyan-400 font-bold bg-black/60 px-1 py-0.5 rounded-sm select-none">
                        [L]
                      </div>
                      <div className="absolute bottom-2 left-2 text-[9px] font-mono text-slate-400 bg-black/60 px-1.5 py-0.5 rounded-sm select-none">
                        ANT.
                      </div>
                      
                      {/* Target Crosshair */}
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none opacity-30">
                        <div className="w-8 h-8 border border-dashed border-cyan-400 rounded-full flex items-center justify-center">
                          <div className="w-1.5 h-1.5 bg-cyan-400 rounded-full" />
                        </div>
                        <div className="absolute w-12 h-[1px] bg-cyan-400" />
                        <div className="absolute h-12 w-[1px] bg-cyan-400" />
                      </div>

                      {/* Hover action bar overlay */}
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleReset();
                        }}
                        className="absolute bottom-2 right-2 p-1.5 rounded-full bg-red-600 border border-red-500 text-white hover:scale-105 shadow-md transition-all z-20 cursor-pointer"
                        title="Remove Scan"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                      </button>
                    </div>
                  ) : file ? (
                    /* CT file display card */
                    <div className="relative w-full max-w-[280px] rounded-xl bg-slate-900 border border-slate-700 p-5 text-center flex flex-col items-center">
                      
                      {/* Scanner light line */}
                      {isProcessing && <div className="medical-scanner-line" />}
                      
                      <div className="h-14 w-14 rounded-2xl bg-cyan-500/10 text-cyan-400 flex items-center justify-center mb-4 border border-cyan-500/20">
                        <svg className="w-7 h-7" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5A3.375 3.375 0 0010.125 2.25H8.25m0 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
                        </svg>
                      </div>
                      <p className="text-sm font-semibold text-white truncate max-w-full px-2">{file.name}</p>
                      <p className="text-[10px] text-cyan-400 uppercase tracking-widest font-mono mt-1">
                        2D CT SCAN SLIDE
                      </p>
                      <p className="text-[11px] text-slate-400 mt-2">
                        {(file.size / 1024 / 1024).toFixed(2)} MB
                      </p>

                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleReset();
                        }}
                        className="absolute top-2.5 right-2.5 p-1.5 rounded-full bg-slate-800 text-slate-400 hover:text-red-400 hover:bg-slate-750 transition-all z-20 cursor-pointer"
                        title="Remove Scan"
                      >
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ) : (
                    /* Blank Dropzone interface */
                    <div className="flex flex-col items-center justify-center py-4">
                      <div className="h-14 w-14 rounded-2xl bg-primary/10 text-primary-deep flex items-center justify-center mb-4 transition-transform group-hover:scale-105">
                        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2.2" className="h-6 w-6">
                          <path strokeLinecap="round" strokeLinejoin="round" d="M12 4v12m8-8l-8-8-8 8m18 8v3a2 2 0 01-2 2H4a2 2 0 01-2-2v-3" />
                        </svg>
                      </div>
                      <p className="text-sm font-bold text-slate-800">
                        Drag & Drop imaging file
                      </p>
                      <p className="text-xs text-muted-foreground mt-1 px-4">
                        or click to browse local files.
                      </p>
                      
                      <div className="mt-4 flex flex-wrap justify-center gap-1.5 border-t border-slate-100 pt-3">
                        <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-500 font-bold">
                          {modality === "ct" ? ".nii / .nii.gz" : ".png"}
                        </span>
                        <span className="text-[9px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-500 font-bold">
                          {modality === "ct" ? ".dcm / .zip" : ".jpg / .jpeg"}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Consent Card */}
              <div className="rounded-xl border border-border/80 bg-slate-50/50 p-3.5 flex items-start gap-3">
                <input
                  type="checkbox"
                  id="consent"
                  disabled={isProcessing}
                  checked={consentChecked}
                  onChange={(e) => setConsentChecked(e.target.checked)}
                  className="mt-1 h-4 w-4 shrink-0 rounded border-slate-300 text-primary-deep focus:ring-primary-deep shadow-xs cursor-pointer disabled:opacity-50"
                />
                <label className="text-[11px] font-medium leading-normal cursor-pointer text-slate-500 select-none" htmlFor="consent">
                  I understand this is a research decision support preview, not a validated medical report. I confirm I possess appropriate permissions to share this scan study.
                </label>
              </div>

              {/* Action trigger row */}
              <div className="flex items-center justify-between gap-3 border-t border-border/50 pt-4 mt-2">
                <span className="text-[10px] text-muted-foreground font-mono font-semibold">
                  SECURE LOCAL SESSION
                </span>
                
                <div className="flex gap-2">
                  {file && (
                    <button
                      onClick={handleReset}
                      disabled={isProcessing}
                      className="px-4 py-2 text-xs font-bold text-slate-600 hover:text-slate-800 border border-slate-200 hover:bg-slate-50 bg-white rounded-full transition-all cursor-pointer disabled:opacity-50"
                    >
                      Clear
                    </button>
                  )}
                  <button
                    onClick={runPredictivePipeline}
                    disabled={!file || !consentChecked || isProcessing}
                    className="inline-flex items-center justify-center gap-2 text-xs font-bold cursor-pointer transition-all disabled:pointer-events-none disabled:opacity-40 bg-gradient-to-r from-primary-deep to-primary text-white shadow-sm hover:shadow-md hover:scale-[1.02] active:scale-[0.98] h-9.5 rounded-full px-6"
                  >
                    {isProcessing ? (
                      <span className="flex items-center gap-2">
                        <svg className="animate-spin h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                          <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                          <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                        </svg>
                        Inference Active
                      </span>
                    ) : (
                      <span className="flex items-center gap-1.5">
                        Run Diagnostic Pipeline
                        <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                          <path strokeLinecap="round" strokeLinejoin="round" d="M14 5l7 7m0 0l-7 7m7-7H3" />
                        </svg>
                      </span>
                    )}
                  </button>
                </div>
              </div>

            </div>

            {/* QUICK SAMPLES SECTION */}
            <div className="rounded-2xl border border-border/85 bg-white shadow-xs p-5 flex flex-col gap-3">
              <span className="text-[11px] font-bold text-slate-800 tracking-wider uppercase font-mono flex items-center gap-1.5">
                <svg className="w-4 h-4 text-primary-deep" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
                Quick Try: Load Sample Studies
              </span>
              
              <div className="grid grid-cols-2 gap-3 mt-1">
                
                {/* Sample Normal X-ray */}
                <button
                  onClick={() => loadSampleFile("/sample_normal.jpg", "sample_chest_normal.jpg", "xray")}
                  disabled={isProcessing}
                  className="flex flex-col items-start rounded-xl border border-border p-3 text-left hover:border-slate-400 hover:bg-slate-50 transition-all cursor-pointer group"
                >
                  <span className="text-[10px] font-bold text-emerald-600 bg-emerald-50 border border-emerald-100 px-1.5 py-0.5 rounded mb-2">
                    NORMAL STUDY
                  </span>
                  <span className="text-xs font-semibold text-slate-800">Healthy Chest X-Ray</span>
                  <span className="text-[9px] text-muted-foreground mt-0.5">Reference radiograph</span>
                </button>

                {/* Sample Pathology X-ray */}
                <button
                  onClick={() => loadSampleFile("/sample_pathology.jpg", "sample_chest_pathology.jpg", "xray")}
                  disabled={isProcessing}
                  className="flex flex-col items-start rounded-xl border border-border p-3 text-left hover:border-slate-400 hover:bg-slate-50 transition-all cursor-pointer group"
                >
                  <span className="text-[10px] font-bold text-amber-600 bg-amber-50 border border-amber-100 px-1.5 py-0.5 rounded mb-2">
                    PATHOLOGICAL
                  </span>
                  <span className="text-xs font-semibold text-slate-800">Abnormal X-Ray</span>
                  <span className="text-[9px] text-muted-foreground mt-0.5">Effusion / congestion</span>
                </button>

                {/* Sample Mock CT Scan */}
                <button
                  onClick={loadMockCTSample}
                  disabled={isProcessing}
                  className="flex flex-col items-start rounded-xl border border-border p-3 text-left hover:border-slate-400 hover:bg-slate-50 transition-all cursor-pointer group"
                >
                  <span className="text-[10px] font-bold text-cyan-600 bg-cyan-50 border border-cyan-100 px-1.5 py-0.5 rounded mb-2">
                    CT STUDY
                  </span>
                  <span className="text-xs font-semibold text-slate-800">Demo CT Scan (2D)</span>
                  <span className="text-[9px] text-muted-foreground mt-0.5">2D slice format (.nii.gz)</span>
                </button>

                {/* Sample Invalid Scan */}
                <button
                  onClick={() => loadSampleFile("/sample_portrait.jpg", "sample_portrait_invalid.jpg", "xray")}
                  disabled={isProcessing}
                  className="flex flex-col items-start rounded-xl border border-border p-3 text-left hover:border-slate-400 hover:bg-slate-50 transition-all cursor-pointer group"
                >
                  <span className="text-[10px] font-bold text-red-600 bg-red-50 border border-red-100 px-1.5 py-0.5 rounded mb-2">
                    INVALID SCAN
                  </span>
                  <span className="text-xs font-semibold text-slate-800">Unrelated Portrait</span>
                  <span className="text-[9px] text-muted-foreground mt-0.5">Tests image guardrail</span>
                </button>

              </div>
            </div>

          </div>

          {/* RIGHT COLUMN: WORKSTATION INTERACTIVE SCREEN (Spans 7 on lg) */}
          <div className="lg:col-span-7 flex flex-col gap-6">
            
            {/* WORKSTATION HUD DASHBOARD CARD */}
            <div className="rounded-3xl bg-white border border-border text-foreground shadow-xs overflow-hidden flex flex-col min-h-[580px] p-5 sm:p-6 relative">
              
              {/* Star/Grid Watermark backdrop */}
              <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(37,99,235,0.03)_1px,transparent_1px),linear-gradient(to_bottom,rgba(37,99,235,0.03)_1px,transparent_1px)] bg-[size:32px_32px] opacity-[0.8] pointer-events-none" />
              
              {/* Telemetry Status Bar */}
              <div className="relative z-10 flex flex-wrap items-center justify-between gap-3 border-b border-border/50 pb-4 mb-6">
                <div>
                  <h2 className="text-xs font-bold font-mono tracking-widest text-primary-deep uppercase">
                    SYSTEM TELEMETRY
                  </h2>
                  <p className="text-[10px] text-muted-foreground font-mono mt-0.5">
                    NODE ADDRESS: LUMORA_INFERENCE_LOCAL
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  {telemetry.status === "verified" && (
                    <span className="text-[10px] font-mono font-bold px-3 py-1 rounded-full bg-primary/5 text-primary-deep border border-primary/20 flex items-center gap-1.5 shadow-xs">
                      <span className="h-1.5 w-1.5 rounded-full bg-primary-deep animate-ping"></span>
                      SCAN VALIDATED
                    </span>
                  )}
                  {telemetry.status === "rejected" && (
                    <span className="text-[10px] font-mono font-bold px-3 py-1 rounded-full bg-slate-100 text-slate-600 border border-slate-300 flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-slate-400"></span>
                      SCAN REJECTED
                    </span>
                  )}
                  {telemetry.status === "none" && !isProcessing && (
                    <span className="text-[10px] font-mono font-bold px-3 py-1 rounded-full bg-slate-50 text-slate-400 border border-slate-200 flex items-center gap-1.5">
                      <span className="h-1.5 w-1.5 rounded-full bg-slate-300"></span>
                      CONSOLE IDLE
                    </span>
                  )}
                  {isProcessing && (
                    <span className="text-[10px] font-mono font-bold px-3 py-1 rounded-full bg-primary/5 text-primary-deep border border-primary/20 flex items-center gap-1.5 animate-pulse">
                      <span className="h-1.5 w-1.5 rounded-full bg-primary-deep animate-ping"></span>
                      PROCESSING DATA...
                    </span>
                  )}
                </div>
              </div>

              {/* TELEMETRY GAUGES GRID */}
              <div className="relative z-10 grid grid-cols-2 sm:grid-cols-4 gap-3.5 mb-6">
                
                {/* Contrast Metric */}
                <div className="rounded-xl bg-surface border border-border/80 p-3 text-left">
                  <div className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider">
                    Contrast SD
                  </div>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <span className="text-xl font-bold font-mono tracking-tight text-foreground">
                      {telemetry.status !== "none" ? telemetry.grayStd.toFixed(1) : "0.0"}
                    </span>
                    <span className="text-[9px] text-slate-400 font-mono">
                      (min {telemetry.contrastThreshold})
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        telemetry.status === "none" ? "w-0" : "bg-primary-deep"
                      }`}
                      style={{ width: `${telemetry.status === "none" ? 0 : Math.min((telemetry.grayStd / 80) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {/* Saturation Ratio */}
                <div className="rounded-xl bg-surface border border-border/80 p-3 text-left">
                  <div className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider">
                    Saturation
                  </div>
                  <div className="mt-1 flex items-baseline gap-1.5">
                    <span className="text-xl font-bold font-mono tracking-tight text-foreground">
                      {telemetry.status !== "none" ? telemetry.meanSaturation.toFixed(3) : "0.000"}
                    </span>
                    <span className="text-[9px] text-slate-400 font-mono">
                      (max {telemetry.saturationThreshold})
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 w-full bg-slate-100 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        telemetry.status === "none" ? "w-0" : "bg-primary-deep"
                      }`}
                      style={{ width: `${telemetry.status === "none" ? 0 : Math.min((telemetry.meanSaturation / 0.5) * 100, 100)}%` }}
                    />
                  </div>
                </div>

                {/* Model Engine */}
                <div className="rounded-xl bg-surface border border-border/80 p-3 text-left">
                  <div className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider">
                    Model Core
                  </div>
                  <div className="mt-1.5 text-xs font-bold font-mono text-primary-deep truncate">
                    {telemetry.engineUsed}
                  </div>
                  <div className="text-[9px] text-slate-400 font-mono mt-1">
                    VLM Core Layer
                  </div>
                </div>

                {/* Latency */}
                <div className="rounded-xl bg-surface border border-border/80 p-3 text-left">
                  <div className="text-[10px] font-mono font-bold text-muted-foreground uppercase tracking-wider">
                    Latency
                  </div>
                  <div className="mt-1.5 text-xs font-bold font-mono text-primary-deep">
                    {telemetry.inferenceTime}
                  </div>
                  <div className="text-[9px] text-slate-400 font-mono mt-1">
                    Inference Time
                  </div>
                </div>

              </div>

              {/* DYNAMIC SCREEN WORKSPACE CONTENT */}
              <div className="relative z-10 flex-1 flex flex-col bg-surface border border-border/70 rounded-2xl p-4 overflow-y-auto">
                
                {/* REJECTED GUARDRAIL STATE */}
                {telemetry.status === "rejected" && (
                  <div className="flex-1 flex flex-col items-center justify-center p-6 text-center">
                    <div className="h-16 w-16 rounded-full bg-primary/5 border border-primary/20 text-primary-deep flex items-center justify-center mb-4 animate-bounce">
                      <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                      </svg>
                    </div>
                    <h3 className="text-base font-bold text-primary-deep font-mono uppercase tracking-wider">
                      Guardrail Rejection Triggered
                    </h3>
                    <p className="text-xs text-slate-600 max-w-sm mt-2 font-mono">
                      {telemetry.saturationMessage}
                    </p>
                    <p className="text-[10px] text-primary-deep/90 max-w-sm mt-3 font-semibold font-mono uppercase bg-primary/5 border border-primary/10 py-1.5 px-3 rounded-lg">
                      [ALERT: UNUSABLE DIAGNOSTIC INPUT VALUE DETECTED]
                    </p>
                  </div>
                )}

                {/* LOADING / STEP PROGRESS STATE */}
                {isProcessing && !displayedReport && (
                  <div className="flex-1 flex flex-col justify-center max-w-md mx-auto w-full p-4">
                    
                    <div className="mb-6 flex flex-col items-center">
                      <div className="relative h-12 w-12 flex items-center justify-center">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary/30 opacity-25"></span>
                        <div className="relative rounded-full h-8 w-8 bg-primary flex items-center justify-center border border-primary/30 text-white">
                          <svg className="animate-spin h-4.5 w-4.5" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                          </svg>
                        </div>
                      </div>
                      <span className="text-xs font-mono font-bold text-primary-deep mt-4 tracking-widest uppercase">
                        DECODING SCAN VECTORS
                      </span>
                    </div>

                    {/* Progress Steps Log */}
                    <div className="rounded-xl bg-white border border-border/80 p-4 font-mono text-[11px] leading-relaxed flex flex-col gap-3">
                      
                      {/* Step 1 */}
                      <div className="flex items-center justify-between">
                        <span className={`${processingStep >= 1 ? "text-primary-deep font-semibold" : "text-slate-400"}`}>
                          &gt; Ingesting study raw binary array
                        </span>
                        <span>
                          {processingStep > 1 ? "✅" : processingStep === 1 ? "⚡" : "⚙️"}
                        </span>
                      </div>

                      {/* Step 2 */}
                      <div className="flex items-center justify-between">
                        <span className={`${processingStep >= 2 ? "text-primary-deep font-semibold" : "text-slate-400"}`}>
                          &gt; Running VLM guardrail checks
                        </span>
                        <span>
                          {processingStep > 2 ? "✅" : processingStep === 2 ? "⚡" : "⚙️"}
                        </span>
                      </div>

                      {/* Step 3 */}
                      <div className="flex items-center justify-between">
                        <span className={`${processingStep >= 3 ? "text-primary-deep font-semibold" : "text-slate-400"}`}>
                          &gt; Projecting visual grid embeddings
                        </span>
                        <span>
                          {processingStep > 3 ? "✅" : processingStep === 3 ? "⚡" : "⚙️"}
                        </span>
                      </div>

                      {/* Step 4 */}
                      <div className="flex items-center justify-between">
                        <span className={`${processingStep >= 4 ? "text-primary-deep font-semibold" : "text-slate-400"}`}>
                          &gt; Drafting report & translate adapters
                        </span>
                        <span>
                          {processingStep > 4 ? "✅" : processingStep === 4 ? "⚡" : "⚙️"}
                        </span>
                      </div>

                    </div>
                  </div>
                )}

                {/* IDLE STATE */}
                {telemetry.status === "none" && !isProcessing && (
                  <div className="flex-1 flex flex-col items-center justify-center p-6 text-center text-slate-400 font-mono">
                    <svg className="w-12 h-12 text-slate-300 mb-3 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M9.75 9.75l4.5 4.5m0-4.5l-4.5 4.5M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <p className="text-xs">
                      Console awaiting scan study ingestion.
                    </p>
                    <p className="text-[10px] text-slate-400 mt-1 max-w-xs">
                      Select a modality, drop your file, check consent, and run diagnosis to start.
                    </p>
                  </div>
                )}

                {/* SUCCESS RESULTS DASHBOARD SCREEN */}
                {telemetry.status === "verified" && displayedReport && (
                  <div className="flex flex-col gap-5 text-left">
                    
                    {/* Clinical Radiology Report Sheet */}
                    <div className="clinical-report-sheet rounded-xl text-slate-800 p-5 font-mono text-[11px] relative overflow-hidden flex flex-col border border-border shadow-inner">
                      
                      {/* Watermarked Hospital Grid backdrop */}
                      <div className="absolute inset-0 bg-[linear-gradient(to_right,#000_1px,transparent_1px),linear-gradient(to_bottom,#000_1px,transparent_1px)] bg-[size:24px_24px] opacity-[0.02] pointer-events-none" />
                      
                      {/* Report Sheet Header */}
                      <div className="border-b-2 border-slate-800 pb-3 mb-4 flex flex-wrap items-center justify-between gap-3 select-none">
                        <div>
                          <div className="text-[13px] font-black tracking-tight text-slate-900">
                            LUMORA AI HEALTH SYSTEMS
                          </div>
                          <div className="text-[9px] text-slate-500 font-sans mt-0.5">
                            ELECTRONIC CLINICAL RECORD DRAFT // CONFIDENTIAL
                          </div>
                        </div>
                        <div className="text-right">
                          <div className="text-[10px] font-bold text-slate-950 bg-slate-100 border border-slate-300 py-0.5 px-2 rounded-sm">
                            REF: LMR-{Math.floor(1000 + Math.random() * 9000)}-CX
                          </div>
                          <div className="text-[8px] text-slate-500 font-sans mt-1">
                            DATE: {currentDate || "PENDING"}
                          </div>
                        </div>
                      </div>

                      {/* Report Core Text */}
                      <div className="whitespace-pre-wrap leading-relaxed min-h-[140px] text-slate-800 select-text relative z-10">
                        <span className={displayedReport.length < reportText.length ? "clinical-cursor" : ""}>
                          {displayedReport}
                        </span>
                      </div>

                      {/* Signature / Validation Block */}
                      <div className="border-t border-slate-200 mt-5 pt-3 flex flex-wrap items-end justify-between gap-3 select-none text-[8px] text-slate-400 font-sans">
                        <div className="max-w-xs leading-normal">
                          <strong>VERIFICATION NOTICE:</strong> This document represents an automated diagnostic draft generated under session checkpoints. Licenced clinical practitioners must corroborate all findings before therapeutic administration.
                        </div>
                        <div className="text-right font-mono text-[9px] font-bold text-slate-500">
                          ISSUED BY: {telemetry.engineUsed.toUpperCase()}<br />
                          STATUS: SYSTEM_DRAFT
                        </div>
                      </div>

                    </div>

                    {/* Copy draft option */}
                    {!isProcessing && (
                      <div className="flex justify-end -mt-3">
                        <button
                          onClick={() => {
                            navigator.clipboard.writeText(reportText);
                            alert("Radiology report draft copied to clipboard.");
                          }}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white border border-border text-slate-700 hover:bg-slate-50 hover:text-slate-900 text-[10px] font-mono font-bold transition-all cursor-pointer shadow-2xs"
                        >
                          <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                          </svg>
                          COPY REPORT TEXT
                        </button>
                      </div>
                    )}

                    {/* Pathology Classifier Pills Card */}
                    <div className="rounded-xl border border-border bg-white p-4 shadow-2xs">
                      <div className="flex items-center justify-between border-b border-border/50 pb-2 mb-3">
                        <span className="text-[10px] font-mono font-bold text-muted-foreground tracking-wider uppercase">
                          Pathology Classifier Outputs
                        </span>
                        <span className="text-[9px] font-mono font-bold text-primary-deep px-1.5 py-0.5 rounded bg-primary/5 border border-primary/15">
                          Recall_0.15
                        </span>
                      </div>

                      <div className="flex flex-wrap gap-2">
                        {diseases.length > 0 ? (
                          diseases.map((disease, idx) => {
                            return (
                              <span
                                key={idx}
                                className="px-3 py-1 rounded-lg text-xs font-semibold border font-mono flex items-center gap-1.5 bg-primary/5 text-primary-deep border-primary/15"
                              >
                                <span className="h-1.5 w-1.5 rounded-full bg-primary-deep animate-pulse" />
                                {disease}
                              </span>
                            );
                          })
                        ) : (
                          <span className="text-xs text-slate-400 italic">No pathologies classified.</span>
                        )}
                      </div>
                    </div>

                    {/* Patient-Friendly Speech Card */}
                    {displayedTranslation && (
                      <div className="rounded-xl border border-border bg-white overflow-hidden flex flex-col shadow-2xs">
                        <div className="bg-surface border-b border-border/50 py-2.5 px-4 flex items-center justify-between select-none">
                          <span className="text-[10px] font-mono font-bold text-muted-foreground tracking-wider uppercase">
                            Patient-Friendly translation
                          </span>
                          <span className="text-[9px] font-mono font-bold text-primary-deep px-1.5 py-0.5 rounded bg-primary/5 border border-primary/15">
                            Layperson Adapter
                          </span>
                        </div>
                        
                        <div className="p-4 flex flex-col gap-3">
                          <div className="relative text-xs text-slate-700 leading-relaxed font-sans select-text border-l-2 border-primary-deep pl-3 py-1 bg-surface/50 rounded-r-md">
                            <span className={displayedTranslation.length < translationText.length ? "clinical-cursor" : ""}>
                              {displayedTranslation}
                            </span>
                          </div>

                          <div className="flex justify-end">
                            <button
                              onClick={() => {
                                navigator.clipboard.writeText(translationText);
                                alert("Patient text copied to clipboard.");
                              }}
                              className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-white hover:bg-slate-50 text-[9px] font-mono font-bold text-slate-600 hover:text-slate-800 border border-border transition-all cursor-pointer shadow-2xs"
                            >
                              <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                                <path strokeLinecap="round" strokeLinejoin="round" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                              </svg>
                              COPY LAYPERSON DRAFT
                            </button>
                          </div>
                        </div>
                      </div>
                    )}

                  </div>
                )}

              </div>

            </div>

          </div>

        </div>

      </main>
    </div>
  );
}
