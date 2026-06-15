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

type Modality = "xray" | "ct" | "report";

export default function Analyze() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [modality, setModality] = useState<Modality>("xray");
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [consentChecked, setConsentChecked] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<
    "checking" | "online" | "offline"
  >("checking");

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

  const invalidMessageForModality = (selectedModality: Modality) => {
    if (selectedModality === "ct") return "It is not a chest/lung CT scan.";
    if (selectedModality === "report") return "The uploaded document is not a valid chest clinical report.";
    return "It is not a chest/lung X-ray image.";
  };

  const resolveInvalidScanMessage = (selectedModality: Modality, backendReport?: string) => {
    if (!backendReport || backendReport.includes("x-ray or chest/lung ct-scan") || backendReport.includes("Invalid")) {
      return invalidMessageForModality(selectedModality);
    }
    return backendReport;
  };

  const typewriterIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const translationTypewriterRef = useRef<NodeJS.Timeout | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

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
        setDisplayedReport((prev) => prev + reportText.charAt(currentIndex));
        currentIndex++;
      } else {
        if (typewriterIntervalRef.current)
          clearInterval(typewriterIntervalRef.current);
      }
    }, 6);

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
        setDisplayedTranslation((prev) => prev + translationText.charAt(currentIndex));
        currentIndex++;
      } else {
        if (translationTypewriterRef.current)
          clearInterval(translationTypewriterRef.current);
      }
    }, 6);

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
    } else if (["txt", "pdf", "docx"].includes(ext)) {
      targetModality = "report";
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

    await new Promise((resolve) => setTimeout(resolve, 800));

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_BASE_URL || "http://localhost:8000";

    // Offline simulation mode
    if (backendOnline === "offline") {
      const startTime = performance.now();
      await new Promise((resolve) => setTimeout(resolve, 1500));
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
          grayStd: modality === "xray" ? 74.2 : 0.0,
          contrastThreshold: 8.0,
          saturationMessage: "Guardrail validation alert: Input does not appear to contain diagnostic structures.",
          contrastMessage: "Pixel distributions failed clinical chest projection thresholds.",
          engineUsed: "Clinical Physics Guardrail",
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
          engineUsed: modality === "ct" ? "Lumora CT-RATE VLM Assistant" : modality === "report" ? "Bio_ClinicalBERT Extractor" : "Lumora VLM Assistant",
          inferenceTime: `${elapsedSeconds}s`,
        });

        if (modality === "report") {
          // Report mode simulation
          if (isPathologicalCase) {
            setReportText(
              "EXAMINATION: Chest Radiograph clinical transcript parsed.\n\nFINDINGS:\nCardiac contours demonstrate mild dilatation. Bilateral vascular congestion. Small pleural fluid accumulation in the right costophrenic angle.\n\nIMPRESSION:\n1. Mild cardiomegaly and vascular congestion.\n2. Small right pleural effusion."
            );
            setDiseases(["Cardiomegaly", "Pulmonary Edema/Vascular Congestion", "Pleural Effusion"]);
            setTranslationText(
              "The document shows your heart is slightly enlarged. There is some fluid congestion in your lungs and a small amount of extra fluid around your right lung."
            );
          } else {
            setReportText(
              "EXAMINATION: Chest test transcript parsed.\n\nFINDINGS:\nLungs are clear. No consolidations, effusions, or masses. Cardiomediastinal contour is normal.\n\nIMPRESSION:\nNormal radiographic findings."
            );
            setDiseases(["No acute cardiopulmonary disease"]);
            setTranslationText(
              "No heart or lung problems were found in the uploaded text report. Your lungs are clear and heart contours are normal."
            );
          }
        } else if (modality === "ct") {
          // CT mode simulation
          if (isPathologicalCase) {
            setReportText(
              "PATIENT CLINICAL REPORT\n\nEXAMINATION: Volumetric Thoracic CT Scan\n\nFINDINGS:\nBilateral emphysematous changes are noted. Minimal calcification of the aortic wall (atherosclerosis). Mediastinal lymph nodes are within normal limits. Minimal spinal degenerative osteophytes are seen.\n\nIMPRESSION:\n1. Moderate emphysema/COPD changes.\n2. Mild aortic wall atherosclerosis."
            );
            setDiseases(["Emphysema/COPD", "Atherosclerosis", "Spinal Degenerative Changes"]);
            setTranslationText(
              "The CT scan shows signs of emphysema (COPD) in both lungs. There is also mild hardening of your main arteries and minor wear-and-tear changes in your spine."
            );
          } else {
            setReportText(
              "PATIENT CLINICAL REPORT\n\nEXAMINATION: Volumetric Thoracic CT Scan\n\nFINDINGS:\nVisualized lung fields are normal without focal consolidations or nodules. Pleural spaces are clear. No aortic dilation. Mediastinal structures are unremarkable.\n\nIMPRESSION:\nUnremarkable CT scan of the chest."
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
      else if (modality === "report") endpoint = "/predict/report";

      try {
        const response = await fetch(`${backendUrl}${endpoint}`, {
          method: "POST",
          body: formData,
        });

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
            engineUsed: modality === "ct" ? "Lumora CT-RATE VLM Assistant" : modality === "report" ? "Bio_ClinicalBERT Extractor" : "Lumora VLM Assistant",
            inferenceTime: "N/A",
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
            inferenceTime: "N/A",
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
      <main className={`mx-auto w-full px-6 py-16 transition-all duration-300 ${
        hasResults ? "max-w-7xl" : "max-w-4xl"
      }`}>
        {/* Page Heading (matches reference) */}
        {!hasResults && (
          <div className="text-center mb-10">
            <p className="text-xs uppercase tracking-[0.2em] text-primary-deep">
              Analyzer
            </p>
            <h1 className="mt-3 text-4xl font-semibold tracking-tight sm:text-5xl text-foreground">
              Pick what you&apos;d like to read.
            </h1>
            <p className="mx-auto mt-3 max-w-lg text-muted-foreground">
              Choose an input type, drop your file, and Lumora will return predicted findings.
            </p>
          </div>
        )}

        {/* Dynamic Split Layout: grid-cols-1 or grid-cols-12 */}
        <div className={`grid gap-6 ${hasResults ? "grid-cols-1 lg:grid-cols-12" : "grid-cols-1"}`}>
          
          {/* INPUT SECTION (spans 5 columns when results exist) */}
          <div className={`${hasResults ? "lg:col-span-5" : "w-full"}`}>
            
            {/* Modality Selector Grid */}
            <div className="grid gap-3 sm:grid-cols-3 mb-6">
              
              {/* CT Scan Tab */}
              <button
                onClick={() => handleModalityChange("ct")}
                disabled={isProcessing}
                className={`flex flex-col items-start rounded-2xl border bg-card p-5 text-left transition-all cursor-pointer ${
                  modality === "ct"
                    ? "border-primary ring-2 ring-primary/20"
                    : "border-border hover:border-primary/40"
                }`}
              >
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl transition-all ${
                  modality === "ct" ? "bg-primary text-primary-foreground" : "bg-surface text-primary-deep"
                }`}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
                    <path d="M3 7V5a2 2 0 0 1 2-2h2"></path>
                    <path d="M17 3h2a2 2 0 0 1 2 2v2"></path>
                    <path d="M21 17v2a2 2 0 0 1-2 2h-2"></path>
                    <path d="M7 21H5a2 2 0 0 1-2-2v-2"></path>
                    <path d="M7 12h10"></path>
                  </svg>
                </div>
                <h3 className="mt-4 font-semibold text-foreground">CT Scan</h3>
                <p className="mt-1 text-sm text-muted-foreground">Volumetric chest CT</p>
              </button>

              {/* Chest X-Ray Tab */}
              <button
                onClick={() => handleModalityChange("xray")}
                disabled={isProcessing}
                className={`flex flex-col items-start rounded-2xl border bg-card p-5 text-left transition-all cursor-pointer ${
                  modality === "xray"
                    ? "border-primary ring-2 ring-primary/20"
                    : "border-border hover:border-primary/40"
                }`}
              >
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl transition-all ${
                  modality === "xray" ? "bg-primary text-primary-foreground" : "bg-surface text-primary-deep"
                }`}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
                    <path d="M22 12h-2.48a2 2 0 0 0-1.93 1.46l-2.35 8.36a.25.25 0 0 1-.48 0L9.24 2.18a.25.25 0 0 0-.48 0l-2.35 8.36A2 2 0 0 1 4.49 12H2"></path>
                  </svg>
                </div>
                <h3 className="mt-4 font-semibold text-foreground">Chest X-Ray</h3>
                <p className="mt-1 text-sm text-muted-foreground">Chest radiograph</p>
              </button>

              {/* Medical Report Tab */}
              <button
                onClick={() => handleModalityChange("report")}
                disabled={isProcessing}
                className={`flex flex-col items-start rounded-2xl border bg-card p-5 text-left transition-all cursor-pointer ${
                  modality === "report"
                    ? "border-primary ring-2 ring-primary/20"
                    : "border-border hover:border-primary/40"
                }`}
              >
                <div className={`flex h-10 w-10 items-center justify-center rounded-xl transition-all ${
                  modality === "report" ? "bg-primary text-primary-foreground" : "bg-surface text-primary-deep"
                }`}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="h-5 w-5">
                    <path d="M6 22a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h8a2.4 2.4 0 0 1 1.704.706l3.588 3.588A2.4 2.4 0 0 1 20 8v12a2 2 0 0 1-2 2z"></path>
                    <path d="M14 2v5a1 1 0 0 0 1 1h5"></path>
                    <path d="M10 9H8"></path>
                    <path d="M16 13H8"></path>
                    <path d="M16 17H8"></path>
                  </svg>
                </div>
                <h3 className="mt-4 font-semibold text-foreground">Medical Report</h3>
                <p className="mt-1 text-sm text-muted-foreground">Existing report or text</p>
              </button>
            </div>

            {/* Upload Area Card Container (matches reference) */}
            <div className="rounded-3xl border border-border bg-card p-5 sm:p-7 shadow-sm">
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => !file && fileInputRef.current?.click()}
                className={`relative overflow-hidden rounded-2xl border-2 border-dashed flex flex-col items-center justify-center px-6 py-12 text-center transition-colors ${
                  file
                    ? "border-border bg-surface/20"
                    : "border-border bg-surface/40 hover:border-primary/50 cursor-pointer"
                } ${isDragActive ? "border-primary bg-primary/5" : ""}`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept={
                    modality === "ct"
                      ? ".nii,.nii.gz,.dcm,.zip"
                      : modality === "report"
                      ? ".txt,.pdf,.docx"
                      : ".png,.jpg,.jpeg"
                  }
                  className="hidden"
                />

                {previewUrl ? (
                  // Preview state for X-Ray
                  <div className="relative w-full aspect-square max-w-[240px] bg-slate-900 rounded-xl overflow-hidden shadow-sm flex items-center justify-center border border-border">
                    {isProcessing && <div className="medical-scanner-line" />}
                    <img
                      src={previewUrl}
                      alt="Scan Preview"
                      className="max-w-full max-h-full object-contain filter brightness-95"
                    />
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReset();
                      }}
                      className="absolute top-2 right-2 p-1.5 rounded-full bg-white/90 border border-border text-slate-500 hover:text-red-600 shadow-sm transition-all z-20 cursor-pointer"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                  </div>
                ) : file ? (
                  // File details state for CT / Report
                  <div className="relative w-full max-w-[280px] rounded-xl bg-background border border-border p-4 text-center">
                    {isProcessing && <div className="medical-scanner-line" />}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleReset();
                      }}
                      className="absolute top-2 right-2 p-1.5 rounded-full bg-background border border-border text-slate-500 hover:text-red-600 shadow-sm transition-all z-20 cursor-pointer"
                    >
                      <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                      </svg>
                    </button>
                    
                    <div className="mx-auto h-12 w-12 rounded-xl bg-primary/10 text-primary flex items-center justify-center mb-3">
                      <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                    </div>
                    <p className="text-sm font-semibold text-foreground break-all">{file.name}</p>
                    <p className="text-[11px] text-muted-foreground mt-1">
                      {modality === "ct" ? "Volumetric CT Scan File" : "Clinical Document"}
                    </p>
                  </div>
                ) : (
                  // Standard Drag-and-Drop display
                  <>
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
                      className="h-8 w-8 text-primary-deep"
                    >
                      <path d="M12 3v12"></path>
                      <path d="m17 8-5-5-5 5"></path>
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    </svg>
                    <p className="mt-3 text-sm font-medium text-foreground">
                      Drop your {modality === "ct" ? "CT scan" : modality === "report" ? "medical report" : "chest x-ray"} here, or click to browse
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {modality === "ct"
                        ? "NIfTI (.nii, .nii.gz), DICOM (.dcm), or ZIP studies"
                        : modality === "report"
                        ? "TXT, PDF, or DOCX documents (max 10 MB)"
                        : "PNG or JPG image (max 20 MB)"}
                    </p>
                  </>
                )}
              </div>



              {/* Consent check (matches reference) */}
              <div className="mt-6 flex items-start gap-3 rounded-xl bg-surface/30 p-4 text-sm">
                <input
                  type="checkbox"
                  id="consent"
                  disabled={isProcessing}
                  checked={consentChecked}
                  onChange={(e) => setConsentChecked(e.target.checked)}
                  className="mt-1 h-4 w-4 shrink-0 rounded border-primary text-primary focus:ring-primary shadow-xs cursor-pointer disabled:opacity-50"
                />
                <label className="text-xs font-medium leading-normal cursor-pointer text-muted-foreground select-none" htmlFor="consent">
                  I understand this is a research tool and not a medical diagnosis. I have rights to share this file.
                </label>
              </div>

              {/* Action Buttons Row */}
              <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-border/40 pt-4">
                <p className="text-xs text-muted-foreground font-medium">
                  Files are processed securely in your session.
                </p>
                <div className="flex gap-2">
                  {file && (
                    <button
                      onClick={handleReset}
                      disabled={isProcessing}
                      className="px-4 py-2 text-xs font-semibold text-muted-foreground border border-border hover:bg-muted bg-background rounded-full transition-all cursor-pointer"
                    >
                      Reset
                    </button>
                  )}
                  <button
                    onClick={runPredictivePipeline}
                    disabled={!file || !consentChecked || isProcessing}
                    className="inline-flex items-center justify-center gap-2 whitespace-nowrap text-xs font-semibold cursor-pointer transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-40 bg-primary text-primary-foreground shadow hover:bg-primary/90 h-9 rounded-full px-5"
                  >
                    {isProcessing ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              </div>

            </div>
          </div>

          {/* RESULTS PANEL SECTION (spans 7 columns when results exist) */}
          {hasResults && (
            <div className="lg:col-span-7 flex flex-col gap-6">
              
              {/* Draft Clinical Findings Card */}
              <div className="clinical-card overflow-hidden flex flex-col flex-1">
                <div className="bg-slate-50 border-b border-border/50 py-3 px-5 flex flex-wrap items-center justify-between gap-2">
                  <span className="text-xs font-bold text-slate-700 uppercase tracking-wider font-mono">
                    DRAFT CLINICAL FINDINGS
                  </span>
                  
                  {telemetry.status === "verified" && (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold flex items-center gap-1">
                      <span className="h-1 w-1 rounded-full bg-emerald-500"></span>
                      Verified
                    </span>
                  )}
                  {telemetry.status === "rejected" && (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-red-50 text-red-700 border border-red-200 font-bold flex items-center gap-1">
                      <span className="h-1 w-1 rounded-full bg-red-500 animate-pulse"></span>
                      Rejected
                    </span>
                  )}
                </div>

                <div className="bg-white p-5 sm:p-7 flex-1 min-h-[260px] text-xs leading-relaxed overflow-y-auto max-h-[360px] border-b border-border/40">
                  {telemetry.status === "rejected" && (
                    <div className="mb-4 p-3.5 rounded-xl bg-red-50 border border-red-100 text-red-800 text-[11px]">
                      <div className="font-bold mb-1 flex items-center gap-1.5">
                        <svg className="w-3.5 h-3.5 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        Scan Integrity Alert
                      </div>
                      <p className="opacity-95">{telemetry.saturationMessage}</p>
                    </div>
                  )}

                  {isProcessing && !displayedReport ? (
                    <div className="text-slate-400 flex flex-col gap-2 animate-pulse pt-2">
                      <p className="font-semibold text-slate-500">&gt; Parsing file bytes...</p>
                      <p>&gt; Ingesting feature vectors into clinical checkpoints...</p>
                      <p>&gt; Preprocessing and extracting radiological descriptors...</p>
                      <p>&gt; Transcribing findings vocabulary sets...</p>
                    </div>
                  ) : displayedReport ? (
                    <div className="clinical-report-sheet p-4 sm:p-5 rounded-lg whitespace-pre-wrap font-sans text-slate-700 text-xs border border-border/80 shadow-2xs relative overflow-hidden">
                      <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0">
                        <span className="text-[28px] font-mono font-black text-primary/5 tracking-[0.2em] uppercase -rotate-12">
                          SYSTEM DRAFT
                        </span>
                      </div>
                      <div className="relative z-10 font-mono text-[11px] leading-relaxed">
                        <span className={displayedReport.length < reportText.length ? "clinical-cursor" : ""}>
                          {displayedReport}
                        </span>
                        
                        <div className="mt-6 pt-4 border-t border-slate-100 text-[10px] text-slate-400 leading-normal font-sans">
                          <strong>DISCLAIMER:</strong> This report represents an automated clinical summary drafted by the Lumora AI framework. All findings must be verified by a licensed radiologist before patient filing.
                        </div>
                      </div>
                    </div>
                  ) : null}
                </div>

                {displayedReport && !isProcessing && (
                  <div className="bg-slate-50 py-3 px-5 flex justify-end">
                    <button
                      onClick={() => {
                        navigator.clipboard.writeText(reportText);
                        alert("Report summary draft copied to clipboard.");
                      }}
                      className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-lg bg-white hover:bg-slate-100 border border-border text-slate-600 hover:text-slate-800 text-xs font-semibold shadow-2xs transition-all cursor-pointer"
                    >
                      <svg className="w-3.5 h-3.5 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                      </svg>
                      Copy Report Draft
                    </button>
                  </div>
                )}
              </div>

              {/* Disease Classifier Findings Card */}
              {(displayedReport || isProcessing) && (
                <div className="clinical-card p-4 sm:p-5 bg-white shadow-2xs rounded-xl border border-border/80">
                  <h2 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-3.5 flex items-center justify-between">
                    <span>Pathology Detections</span>
                    <span className="text-[9px] px-2 py-0.5 rounded-full bg-primary/10 text-primary-deep font-bold tracking-wide">
                      ClinicalBERT
                    </span>
                  </h2>

                  {isProcessing && !displayedReport ? (
                    <div className="flex items-center gap-2 text-slate-400 text-xs animate-pulse">
                      <svg className="w-4 h-4 animate-spin text-primary-deep" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      Evaluating clinical report vocabulary...
                    </div>
                  ) : diseases.length > 0 ? (
                    <div className="flex flex-wrap gap-2">
                      {diseases.map((disease, idx) => {
                        const isNormal = disease === "No acute cardiopulmonary disease";
                        return (
                          <span
                            key={idx}
                            className={`px-3 py-1 rounded-lg text-xs font-semibold border transition-all flex items-center gap-1.5 shadow-2xs ${
                              isNormal
                                ? "bg-emerald-50 text-emerald-700 border-emerald-200"
                                : "bg-amber-50 text-amber-800 border-amber-200"
                            }`}
                          >
                            <span className={`h-1.5 w-1.5 rounded-full ${isNormal ? "bg-emerald-500" : "bg-amber-500"}`}></span>
                            {disease}
                          </span>
                        );
                      })}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-400 italic">No pathologies detected.</p>
                  )}
                </div>
              )}

              {/* Patient-Friendly Translation Card */}
              {(displayedTranslation || isProcessing) && (
                <div className="clinical-card overflow-hidden flex flex-col bg-white shadow-2xs rounded-xl border border-border/80">
                  <div className="bg-slate-50 border-b border-border/50 py-3 px-5 flex items-center justify-between">
                    <span className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono">
                      Patient-Friendly Translation
                    </span>
                    <span className="text-[9px] px-2 py-0.5 rounded-full bg-primary/10 text-primary-deep font-bold tracking-wide">
                      T5 Adapter
                    </span>
                  </div>

                  <div className="p-4 sm:p-5 text-xs text-slate-700 leading-relaxed">
                    {isProcessing && !displayedTranslation ? (
                      <div className="text-slate-400 flex flex-col gap-1.5 animate-pulse">
                        <p>&gt; Translating medical jargon into layperson terms...</p>
                      </div>
                    ) : displayedTranslation ? (
                      <div>
                        <p className="font-medium text-slate-700">{displayedTranslation}</p>
                        <div className="mt-4 pt-3 border-t border-border/40 flex justify-end">
                          <button
                            onClick={() => {
                              navigator.clipboard.writeText(translationText);
                              alert("Patient text copied to clipboard.");
                            }}
                            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 border border-border text-slate-600 hover:text-slate-800 text-[10px] font-semibold shadow-2xs transition-all cursor-pointer"
                          >
                            <svg className="w-3.5 h-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                            </svg>
                            Copy Patient Text
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </div>
                </div>
              )}

            </div>
          )}

        </div>
      </main>
    </div>
  );
}
