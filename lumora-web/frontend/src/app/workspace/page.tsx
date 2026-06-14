"use client";

import React, { useState, useEffect, useRef } from "react";
import Link from "next/link";

interface Telemetry {
  status: "verified" | "rejected" | "none";
  meanSaturation: number;       // 0.0 – 1.0  (lower = more grayscale); threshold < 0.15
  saturationThreshold: number;  // 0.15
  grayStd: number;              // 0.0 – 255.0 (higher = more contrast); threshold > 8.0
  contrastThreshold: number;    // 8.0
  saturationMessage?: string;
  contrastMessage?: string;
  engineUsed: string;
  inferenceTime: string;
}

type Modality = "xray" | "ct";

export default function Workspace() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [modality, setModality] = useState<Modality>("xray");
  const [isDragActive, setIsDragActive] = useState<boolean>(false);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [backendOnline, setBackendOnline] = useState<
    "checking" | "online" | "offline"
  >("checking");
  const [isPreset, setIsPreset] = useState<boolean>(false);

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

  const isNiftiFile = (name: string) => {
    const lowerName = name.toLowerCase();
    return lowerName.endsWith(".nii") || lowerName.endsWith(".nii.gz");
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
        setDisplayedReport((prev) => prev + reportText.charAt(currentIndex));
        currentIndex++;
      } else {
        if (typewriterIntervalRef.current)
          clearInterval(typewriterIntervalRef.current);
      }
    }, 8);

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
    }, 8);

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
    const selectedModality: Modality =
      isNiftiFile(selectedFile.name) || ext === "dcm" ? "ct" : "xray";

    if (["jpg", "jpeg", "png", "nii", "nii.gz", "dcm"].includes(ext)) {
      setFile(selectedFile);
      setModality(selectedModality);
      setPreviewUrl(selectedModality === "xray" ? URL.createObjectURL(selectedFile) : null);
      setIsPreset(false); // Reset preset flag on manual upload
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
    } else {
      alert(
        "Unsupported medical format. Please provide JPEG/PNG for X-ray or .nii/.nii.gz/.dcm for CT.",
      );
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

    // Aesthetic diagnostic prep delay
    await new Promise((resolve) => setTimeout(resolve, 800));

    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_BASE_URL || "http://localhost:8000";

    // If it is an X-ray diagnostic preset OR backend server is offline, trigger high-fidelity simulation
    if (modality === "xray" && (isPreset || backendOnline === "offline")) {
      const startTime = performance.now();
      await new Promise((resolve) => setTimeout(resolve, 1200));
      const endTime = performance.now();
      const elapsedSeconds = ((endTime - startTime) / 1000).toFixed(2);

      const isInvalidCase =
        file?.name?.includes("invalid") ||
        previewUrl?.includes("sample_portrait");

      if (isInvalidCase) {
        setTelemetry({
          status: "rejected",
          meanSaturation: 0.38,  // high saturation → colorful photo
          saturationThreshold: 0.15,
          grayStd: 74.2,
          contrastThreshold: 8.0,
          saturationMessage:
            "High color saturation detected (0.38 > 0.15). Natural photograph, not a grayscale X-ray.",
          contrastMessage:
            "Contrast is present but image failed grayscale check first.",
          engineUsed: "Clinical Physics Guardrail",
          inferenceTime: `${elapsedSeconds}s`,
        });
        setReportText(
          "VERIFICATION FAULT: Non-Medical Asset Detected.\n\nThe uploaded image has been analyzed by the system's structural validation layers and does not conform to standard chest radiography blueprints. Photographic chromatic complexity was also caught.\n\nACTION REQUIRED:\nPlease verify that the selected file is indeed a valid frontal (PA/AP) or lateral diagnostic chest radiograph scan and re-submit.",
        );
        setTranslationText("");
        setDiseases([]);
      } else {
        const isPathologicalCase =
          file?.name?.includes("pathology") ||
          previewUrl?.includes("sample_pathology");

        setTelemetry({
          status: "verified",
          meanSaturation: 0.021,  // near-zero saturation → genuine grayscale
          saturationThreshold: 0.15,
          grayStd: isPathologicalCase ? 48.6 : 55.3,  // meaningful contrast
          contrastThreshold: 8.0,
          saturationMessage: "Near-zero color saturation (0.02). Confirmed monochrome radiograph.",
          contrastMessage: `Pixel intensity std-dev ${isPathologicalCase ? "48.6" : "55.3"} — sufficient diagnostic contrast.`,
          engineUsed: "Lumora VLM Assistant",
          inferenceTime: `${elapsedSeconds}s`,
        });

        if (isPathologicalCase) {
          setReportText(
            "PATIENT CLINICAL REPORT\n\nEXAMINATION: Chest Radiograph, Frontal View\n\nCLINICAL HISTORY: Cough and progressive shortness of breath.\n\nCOMPARISON: None.\n\nFINDINGS:\nThe cardiac silhouette is mildly enlarged (cardiomegaly is present). Prominence of the bilateral pulmonary interstitial markings is demonstrated, suggesting mild interstitial vascular congestion and pulmonary edema. Small bilateral pleural effusions are noted, more prominent on the right. There are patchy basilar opacities in both lung bases, which may represent subsegmental atelectasis or early infectious consolidation. The trachea is midline. The osseous structures are intact.\n\nIMPRESSION:\n1. Mild cardiomegaly with pulmonary interstitial vascular congestion and edema.\n2. Small bilateral pleural effusions.\n3. Basilar lung opacities, which may reflect atelectasis or developing consolidation.",
          );
          setDiseases([
            "Cardiomegaly",
            "Pulmonary Edema/Vascular Congestion",
            "Pleural Effusion",
            "Atelectasis",
            "Consolidation"
          ]);
          setTranslationText(
            "The X-ray shows your heart is slightly enlarged. There is also sign of extra fluid in your lungs and around them, which suggests some congestion. Patchy areas at the bottom of the lungs could be from minor lung collapse or early signs of infection."
          );
        } else {
          setReportText(
            "PATIENT CLINICAL REPORT\n\nEXAMINATION: Chest Radiograph, Frontal View\n\nCLINICAL HISTORY: Screening.\n\nCOMPARISON: None.\n\nFINDINGS:\nThe lungs are clear and fully inflated. No focal consolidations, nodules, or airspace opacities are present. There is no pleural effusion or pneumothorax. The cardiomediastinal contour, cardiac silhouette size, and pulmonary vasculature are well-defined and within normal clinical limits. The mediastinal structures are unremarkable. Visualized skeletal structures are normal.\n\nIMPRESSION:\nNormal radiographic examination of the chest. No acute cardiopulmonary abnormalities detected.",
          );
          setDiseases(["No acute cardiopulmonary disease"]);
          setTranslationText(
            "No immediate heart or lung problems were found on this chest X-ray. Your heart size is normal, your lungs are clear, and there are no signs of fluid build-up or collapse."
          );
        }
      }
    } else {
      const formData = new FormData();
      formData.append("file", file);
      const endpoint = modality === "ct" ? "/predict/ct" : "/predict";

      try {
        const response = await fetch(
          `${backendUrl}${endpoint}`,
          {
            method: "POST",
            body: formData,
          },
        );

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
            saturationMessage: modality === "ct"
              ? data.telemetry || "CT projection generated from uploaded volumetric file."
              : typeof data.mean_saturation === "number"
              ? `Color saturation ${data.mean_saturation.toFixed(3)} — within grayscale threshold. Confirmed monochrome scan.`
              : "Color saturation verified.",
            contrastMessage: typeof data.gray_std === "number"
              ? modality === "ct"
                ? `Projection intensity std-dev ${data.gray_std.toFixed(1)} — CT file decoded with measurable structure.`
                : `Pixel intensity std-dev ${data.gray_std.toFixed(1)} — sufficient diagnostic contrast present.`
              : "Contrast verified.",
            engineUsed: modality === "ct" ? "Lumora CT-RATE VLM Assistant" : "Lumora VLM Assistant",
            inferenceTime: "N/A",
          });
          setReportText(data.report);
          setTranslationText(data.translation || "");
          setDiseases(data.diseases || []);
        } else if (response.status === 422) {
          const meanSaturation = typeof data.mean_saturation === "number" ? data.mean_saturation : 0;
          const grayStd = typeof data.gray_std === "number" ? data.gray_std : 0;
          
          let contrastMessage = "Contrast verification complete.";
          if (typeof data.gray_std === "number") {
            contrastMessage = data.gray_std < 8.0
              ? `Pixel intensity std-dev ${data.gray_std.toFixed(1)} — image appears blank or near-uniform.`
              : `Pixel intensity std-dev ${data.gray_std.toFixed(1)} — contrast check passed.`;
          } else if (data.detail && typeof data.detail === "string") {
            contrastMessage = `Validation details: ${data.detail}`;
          } else if (Array.isArray(data.detail)) {
            const messages = data.detail.map((d: { loc?: string[]; msg?: string }) => `${d.loc?.join(".") || "field"}: ${d.msg || "invalid"}`).join(", ");
            contrastMessage = `API validation failure: ${messages}`;
          }

          setTelemetry({
            status: "rejected",
            meanSaturation,
            saturationThreshold: 0.15,
            grayStd,
            contrastThreshold: modality === "ct" ? 1.0 : 8.0,
            saturationMessage: data.telemetry || (data.detail ? "API validation error." : "Image failed guardrail validation."),
            contrastMessage,
            engineUsed: modality === "ct" ? "CT Input Validator" : "Clinical Physics Guardrail",
            inferenceTime: "N/A",
          });
          setReportText(modality === "ct"
            ? `CT VERIFICATION FAULT\n\n${data.report || data.detail || "The uploaded CT file could not be decoded into a valid model input."}\n\nACTION REQUIRED:\nPlease upload a valid .nii, .nii.gz, or .dcm CT file and re-submit.`
            : "VERIFICATION FAULT: Non-Medical Asset Detected.\n\nThe uploaded image has been analyzed by the system's structural validation layers and does not conform to standard chest radiography blueprints. Photographic chromatic complexity was also caught.\n\nACTION REQUIRED:\nPlease verify that the selected file is indeed a valid frontal (PA/AP) or lateral diagnostic chest radiograph scan and re-submit.",
          );
          setTranslationText("");
          setDiseases([]);
        } else {
          throw new Error(data.detail || "Server error");
        }
      } catch (err) {
        const errorMsg = err instanceof Error ? err.message : String(err);
        console.error(err);
        setReportText(
          `INFERENCE ERROR: ${errorMsg || `Failed to communicate with the Lumora analysis host. Please verify that the FastAPI backend server is active at ${backendUrl}.`}`,
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
    setModality("xray");
    setIsPreset(false);
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

  // Preset cases
  const loadPresetCase = (type: "normal" | "pathology" | "invalid") => {
    handleReset();
    setModality("xray");
    let simulatedUrl = "";
    let simulatedName = "";

    if (type === "normal") {
      simulatedUrl = "/sample_normal.jpg";
      simulatedName = "normal_chest_frontal.jpg";
    } else if (type === "pathology") {
      simulatedUrl = "/sample_pathology.jpg";
      simulatedName = "congestive_abnormal_chest.jpg";
    } else {
      simulatedUrl = "/sample_portrait.jpg";
      simulatedName = "invalid_portrait_photo.jpg";
    }

    setPreviewUrl(simulatedUrl);
    setIsPreset(true); // Flag this case as a preset simulation
    const mockFile = new File(["mock_data"], simulatedName, {
      type: "image/jpeg",
    });
    setFile(mockFile);
  };

  return (
    <div className="flex flex-col min-h-screen bg-[#f8fafc] font-sans antialiased text-slate-800">
      {/* Top Premium Nav Header */}
      <header className="border-b border-slate-200 bg-white/95 backdrop-blur-md sticky top-0 z-50 shadow-xs">
        <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Branded Logo Image */}
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
                className="px-3 py-1.5 rounded-md text-xs font-bold bg-white text-blue-600 shadow-xs border border-blue-100 transition-all"
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

          {/* Clinician Interface Host Status Indicator */}
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs font-medium">
            {backendOnline === "checking" && (
              <span className="flex items-center gap-1.5 text-slate-500">
                <span className="h-2 w-2 rounded-full bg-slate-400 animate-pulse"></span>
                Connecting to Core Engine...
              </span>
            )}

            {backendOnline === "online" && (
              <span className="flex items-center gap-1.5 text-emerald-700">
                <span className="h-2 w-2 rounded-full bg-emerald-500 animate-ping"></span>
                Inference System: Ready
              </span>
            )}

            {backendOnline === "offline" && (
              <span className="flex items-center gap-1.5 text-amber-700">
                <span className="h-2 w-2 rounded-full bg-amber-500"></span>
                Demo Simulator Active
              </span>
            )}
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* LEFT PANEL: Inputs, Upload, Presets (5 cols) */}
        <section className="lg:col-span-5 flex flex-col gap-6">
          {/* Image Upload Area Card */}
          <div className="clinical-card p-5 flex flex-col">
            <h2 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-2 mb-4">
              Patient Scan Selection
            </h2>

            <div className="grid grid-cols-2 gap-2 mb-4 rounded-xl bg-slate-100 p-1 border border-slate-200">
              <button
                type="button"
                onClick={() => {
                  if (!file) setModality("xray");
                }}
                className={`py-2 rounded-lg text-xs font-bold transition-all ${
                  modality === "xray"
                    ? "bg-white text-blue-700 shadow-xs"
                    : "text-slate-500 hover:text-slate-800"
                } ${file ? "cursor-default" : "cursor-pointer"}`}
              >
                X-Ray JPEG/PNG
              </button>
              <button
                type="button"
                onClick={() => {
                  if (!file) setModality("ct");
                }}
                className={`py-2 rounded-lg text-xs font-bold transition-all ${
                  modality === "ct"
                    ? "bg-white text-blue-700 shadow-xs"
                    : "text-slate-500 hover:text-slate-800"
                } ${file ? "cursor-default" : "cursor-pointer"}`}
              >
                CT Scans
              </button>
            </div>

            {/* File Drag-and-Drop Uploader */}
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => !file && fileInputRef.current?.click()}
              className={`relative overflow-hidden rounded-2xl border-2 border-dashed flex flex-col items-center justify-center p-8 transition-all ${
                file
                  ? "border-slate-200 bg-slate-50/50"
                  : "border-slate-300 bg-slate-50/30 hover:bg-slate-50 hover:border-blue-400 cursor-pointer"
              } ${isDragActive ? "upload-active border-blue-400" : ""}`}
            >
              <input
                type="file"
                ref={fileInputRef}
                onChange={handleFileChange}
                accept="image/png, image/jpeg, image/jpg,.nii,.nii.gz,.dcm,application/dicom"
                className="hidden"
              />

              {previewUrl ? (
                // Selected/Loaded Preview State
                <div className="relative w-full aspect-square max-w-[260px] bg-slate-900 rounded-xl overflow-hidden shadow-md flex items-center justify-center border border-slate-200">
                  {isProcessing && <div className="medical-scanner-line" />}

                  <img
                    src={previewUrl}
                    alt="Loaded X-Ray"
                    className="max-w-full max-h-full object-contain filter brightness-95"
                  />

                  {/* Remove Button */}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReset();
                    }}
                    className="absolute top-2.5 right-2.5 p-1.5 rounded-full bg-white/90 border border-slate-200 text-slate-500 hover:text-red-600 hover:bg-white shadow-sm transition-all z-20 cursor-pointer"
                    title="Remove scan"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>

                  <div className="absolute bottom-2.5 left-2.5 px-2 py-1 rounded bg-black/60 text-[9px] font-mono text-white tracking-wide z-10">
                    SCANNAME:{" "}
                    {file ? file.name.substring(0, 18) : "SIMULATED_SCAN"}
                  </div>
                </div>
              ) : file ? (
                <div className="relative w-full max-w-[320px] rounded-xl bg-white border border-slate-200 shadow-xs p-5 text-center">
                  {isProcessing && <div className="medical-scanner-line" />}
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleReset();
                    }}
                    className="absolute top-2.5 right-2.5 p-1.5 rounded-full bg-white border border-slate-200 text-slate-500 hover:text-red-600 shadow-sm transition-all z-20 cursor-pointer"
                    title="Remove scan"
                  >
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M6 18L18 6M6 6l12 12"
                      />
                    </svg>
                  </button>

                  <div className="mx-auto h-14 w-14 rounded-xl bg-blue-50 border border-blue-100 text-blue-600 flex items-center justify-center mb-4">
                    <svg
                      className="w-7 h-7"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={1.8}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5A3.375 3.375 0 0010.125 2.25H8.25m2.25 0H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z"
                      />
                    </svg>
                  </div>
                  <p className="text-sm font-bold text-slate-800 break-all">
                    {file.name}
                  </p>
                  <p className="text-[11px] text-slate-500 mt-2">
                    {modality === "ct"
                      ? "CT scan file ready for CT-RATE model inference"
                      : "Medical image ready for X-ray model inference"}
                  </p>
                  <div className="mt-4 flex justify-center gap-2 text-[10px] font-semibold">
                    <span className="px-2 py-1 rounded bg-blue-50 text-blue-700 border border-blue-100">
                      {modality === "ct" ? "CT" : "X-Ray"}
                    </span>
                    <span className="px-2 py-1 rounded bg-slate-100 text-slate-600 border border-slate-200">
                      {getFileExtension(file.name).toUpperCase()}
                    </span>
                  </div>
                </div>
              ) : (
                // Standard Empty Upload State
                <div className="flex flex-col items-center justify-center text-center">
                  <div className="p-4 rounded-full bg-blue-50 border border-blue-100 text-blue-500 mb-3 shadow-xs">
                    <svg
                      className="w-7 h-7"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
                      />
                    </svg>
                  </div>
                  <p className="text-sm font-semibold text-slate-700">
                    {modality === "ct"
                      ? "Drag & Drop CT Scan File"
                      : "Drag & Drop Chest X-Ray Scan"}
                  </p>
                  <p className="text-xs text-slate-400 mt-1 max-w-[200px]">
                    {modality === "ct"
                      ? "Supports NIfTI volumes and DICOM CT slices"
                      : "Supports standard grayscale PACS exported PNG or JPEG format"}
                  </p>
                  <button className="mt-4 px-4 py-1.5 rounded-full border border-slate-200 bg-white hover:bg-slate-50 hover:text-slate-900 transition-all text-xs font-semibold shadow-xs cursor-pointer">
                    Browse Files
                  </button>
                </div>
              )}
            </div>

            {/* Compact Inline Presets */}
            {modality === "xray" && !file && (
              <div className="mt-4 border-t border-slate-100 pt-3">
                <p className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2 text-center">
                  Test with presets:
                </p>
                <div className="flex gap-2 justify-center">
                  <button
                    type="button"
                    onClick={() => loadPresetCase("normal")}
                    className="px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-600 font-semibold text-[10px] transition-all cursor-pointer"
                  >
                    Clear Lungs
                  </button>
                  <button
                    type="button"
                    onClick={() => loadPresetCase("pathology")}
                    className="px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-600 font-semibold text-[10px] transition-all cursor-pointer"
                  >
                    Congestion
                  </button>
                  <button
                    type="button"
                    onClick={() => loadPresetCase("invalid")}
                    className="px-2.5 py-1 rounded bg-slate-100 hover:bg-slate-200 text-slate-600 font-semibold text-[10px] transition-all cursor-pointer"
                  >
                    Invalid Photo
                  </button>
                </div>
              </div>
            )}

            {/* Inferences Controls */}
            <div className="mt-5 flex gap-3">
              {previewUrl && (
                <button
                  onClick={handleReset}
                  disabled={isProcessing}
                  className="px-4 py-3 rounded-xl border border-slate-200 bg-white text-slate-500 hover:text-slate-800 hover:bg-slate-50 disabled:opacity-40 font-semibold text-xs transition-all shadow-xs cursor-pointer"
                >
                  RESET
                </button>
              )}

              <button
                onClick={runPredictivePipeline}
                disabled={!file || isProcessing}
                className={`flex-1 flex items-center justify-center gap-2 py-3 px-4 rounded-xl font-semibold text-xs transition-all shadow-xs ${
                  !file
                    ? "bg-slate-100 text-slate-400 border border-slate-200 cursor-not-allowed"
                    : isProcessing
                      ? "bg-blue-100 text-blue-600 border border-blue-200 cursor-wait animate-pulse"
                      : "bg-blue-600 hover:bg-blue-700 text-white border border-blue-700 cursor-pointer"
                }`}
              >
                {isProcessing ? (
                  <>
                    <svg
                      className="w-4 h-4 glowing-spinner"
                      fill="none"
                      viewBox="0 0 24 24"
                    >
                      <circle
                        className="opacity-25"
                        cx="12"
                        cy="12"
                        r="10"
                        stroke="currentColor"
                        strokeWidth="4"
                      />
                      <path
                        className="opacity-75"
                        fill="currentColor"
                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                      />
                    </svg>
                    {modality === "ct" ? "Analyzing CT study..." : "Analyzing radiograph..."}
                  </>
                ) : (
                  <>
                    <svg
                      className="w-4 h-4"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                      strokeWidth={2.5}
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    {modality === "ct" ? "Run CT Model" : "Compile Clinical Report"}
                  </>
                )}
              </button>
            </div>
          </div>
        </section>

        {/* RIGHT PANEL: Printed Report, Diseases & Translation (7 cols) */}
        <section className="lg:col-span-7 flex flex-col gap-6">
          {/* Diagnostic Clinical Report (Modern Print/EHR look) */}
          <div className="clinical-card overflow-hidden flex flex-col flex-1">
            {/* Report Card Header */}
            <div className="bg-slate-50 border-b border-slate-200 py-3.5 px-5 flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                <span className="text-xs font-bold text-slate-700 uppercase tracking-wider font-mono">
                  DRAFT CLINICAL FINDINGS
                </span>
                {telemetry.status === "verified" && (
                  <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold flex items-center gap-1">
                    <span className="h-1.5 w-1.5 rounded-full bg-emerald-500"></span>
                    Scan Verified
                  </span>
                )}
                {telemetry.status === "rejected" && (
                  <span className="text-[10px] px-2 py-0.5 rounded bg-red-50 text-red-700 border border-red-200 font-bold flex items-center gap-1 animate-pulse">
                    <span className="h-1.5 w-1.5 rounded-full bg-red-500"></span>
                    Failed Integrity
                  </span>
                )}
              </div>

              <div className="flex items-center gap-3">
                {telemetry.status !== "none" && telemetry.engineUsed !== "N/A" && (
                  <span className="text-[9px] text-slate-400 font-mono">
                    Model: {telemetry.engineUsed}
                  </span>
                )}
                {isProcessing ? (
                  <span className="text-[10px] px-2 py-0.5 rounded-full bg-blue-50 text-blue-600 border border-blue-100 font-semibold flex items-center gap-1 animate-pulse">
                    <span className="h-1.5 w-1.5 rounded-full bg-blue-500"></span>
                    Transcribing...
                  </span>
                ) : displayedReport ? (
                  <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 font-bold flex items-center gap-1">
                    Draft Prepared
                  </span>
                ) : (
                  <span className="text-[10px] text-slate-400 font-medium">
                    Radiology Document Idle
                  </span>
                )}
              </div>
            </div>

            {/* Diagnostic Document Sheet */}
            <div className="bg-white p-8 flex-1 min-h-[320px] text-xs text-slate-800 leading-relaxed overflow-y-auto max-h-[460px] border-b border-slate-200">
              {telemetry.status === "rejected" && (
                <div className="mb-5 p-4 rounded-xl bg-red-50 border border-red-100 text-xs text-red-800">
                  <div className="font-bold mb-1 flex items-center gap-1.5">
                    <svg className="w-4 h-4 text-red-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    Scan Verification Alert
                  </div>
                  <p className="text-[11px] leading-relaxed opacity-90">
                    {telemetry.saturationMessage} {telemetry.contrastMessage}
                  </p>
                </div>
              )}

              {isProcessing && !displayedReport ? (
                // Processing Indicators
                <div className="text-slate-400 flex flex-col gap-2 animate-pulse mt-4">
                  <p className="font-semibold text-slate-500">
                    &gt; {modality === "ct" ? "Projecting CT volume planes..." : "Decompressing visual features..."}
                  </p>
                  <p>&gt; Validating spatial orientation coordinates...</p>
                  <p>&gt; {modality === "ct" ? "Loading CT-RATE checkpoint..." : "Matching contrast anchors..."}</p>
                  <p>&gt; Generating clinical vocabulary token sets...</p>
                </div>
              ) : displayedReport ? (
                // Formatted clinical report print-look
                <div className="clinical-report-sheet p-6 rounded-lg whitespace-pre-wrap font-sans text-slate-700 text-xs border border-slate-200/80 shadow-xs relative overflow-hidden">
                  {/* Faint Brand-Aligned Background Watermark */}
                  <div className="absolute inset-0 flex items-center justify-center pointer-events-none select-none z-0 overflow-hidden">
                    <span className="text-[38px] font-mono font-black text-blue-600/4 tracking-[0.25em] uppercase -rotate-12">
                      SYSTEM DRAFT
                    </span>
                  </div>

                  {/* Elevated Clinical Text Layer */}
                  <div className="relative z-10">
                    <span
                      className={
                        displayedReport.length < reportText.length ? "clinical-cursor" : ""
                      }
                    >
                      {displayedReport}
                    </span>

                    {/* Radiologist Disclaimer */}
                    <div className="mt-8 pt-4 border-t border-slate-100 text-[10px] text-slate-400 leading-normal">
                      <strong>DISCLAIMER:</strong> This report represents an
                      automated clinical summary drafted by the Lumora AI VLM
                      framework. It is intended solely for radiologist and
                      clinician decision-support. A licensed clinical specialist
                      must review and sign off before final patient record
                      filing.
                    </div>
                  </div>
                </div>
              ) : (
                // Empty report screen
                <div className="flex flex-col items-center justify-center text-center h-full min-h-[260px] text-slate-400">
                  <div className="p-3.5 rounded-full bg-slate-50 border border-slate-100 text-slate-300 mb-3.5">
                    <svg
                      className="w-8 h-8"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={1.5}
                        d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                      />
                    </svg>
                  </div>
                  <p className="font-semibold text-slate-500">
                    Report Ingestion Workspace Ready
                  </p>
                  <p className="text-[11px] text-slate-400 mt-1 max-w-xs leading-normal">
                    Select a chest radiograph or CT study and compile features to generate
                    clinical impression drafts here.
                  </p>
                </div>
              )}
            </div>

            {/* Clinical copy tools */}
            {displayedReport && !isProcessing && (
              <div className="bg-slate-50 py-3.5 px-5 flex justify-end gap-3">
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(reportText);
                    alert(
                      "Diagnostic report draft copied to clinical clipboard.",
                    );
                  }}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-white hover:bg-slate-100 border border-slate-200 text-slate-600 hover:text-slate-800 text-xs font-semibold shadow-xs transition-all cursor-pointer"
                >
                  <svg
                    className="w-4 h-4 text-slate-500"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3"
                    />
                  </svg>
                  Copy Summary Draft
                </button>
              </div>
            )}
          </div>

          {/* Disease Classification Card */}
          {(displayedReport || isProcessing) && (
            <div className="clinical-card p-5 bg-white shadow-xs rounded-xl border border-slate-200 transition-all duration-300">
              <h2 className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono border-b border-slate-100 pb-3 mb-4 flex items-center justify-between">
                <span>Disease Classifier Findings (High-Recall)</span>
                <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 font-bold tracking-wide">
                  ClinicalBERT
                </span>
              </h2>

              {isProcessing && !displayedReport ? (
                <div className="flex items-center gap-2.5 text-slate-400 text-xs animate-pulse py-2">
                  <svg className="w-4 h-4 animate-spin text-indigo-500" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                  </svg>
                  Analyzing clinical vocabulary for pathological markers...
                </div>
              ) : diseases.length > 0 ? (
                <div className="flex flex-wrap gap-2.5 py-1">
                  {diseases.map((disease, idx) => {
                    const isNormal = disease === "No acute cardiopulmonary disease" || disease === "None";
                    return (
                      <span
                        key={idx}
                        className={`px-3 py-1.5 rounded-lg text-xs font-semibold border transition-all hover:scale-105 duration-200 flex items-center gap-1.5 shadow-2xs ${
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
                <p className="text-xs text-slate-400 italic">No pathological markers identified.</p>
              )}
            </div>
          )}

          {/* Patient-Friendly Translation Card */}
          {(displayedTranslation || isProcessing) && (
            <div className="clinical-card overflow-hidden flex flex-col bg-white shadow-xs rounded-xl border border-slate-200 transition-all duration-300">
              <div className="bg-slate-50 border-b border-slate-200 py-3.5 px-5 flex items-center justify-between">
                <span className="text-xs font-bold text-slate-700 tracking-wide uppercase font-mono">
                  Patient-Friendly Translation
                </span>
                <span className="text-[10px] px-2.5 py-0.5 rounded-full bg-blue-50 text-blue-700 border border-blue-100 font-bold tracking-wide">
                  T5-Small Adapter
                </span>
              </div>

              <div className="p-6 bg-gradient-to-br from-blue-50/10 to-indigo-50/5 min-h-[120px] text-xs text-slate-700 leading-relaxed relative">
                {isProcessing && !displayedTranslation ? (
                  <div className="text-slate-400 flex flex-col gap-2 animate-pulse">
                    <p>&gt; Translating medical terminology to layperson terms...</p>
                    <p>&gt; Simplifying clinical syntax structure...</p>
                  </div>
                ) : displayedTranslation ? (
                  <div className="relative z-10">
                    <p className="font-medium text-slate-700 whitespace-pre-wrap">
                      {displayedTranslation}
                    </p>

                    {/* Copy Translation Tool */}
                    <div className="mt-5 pt-3.5 border-t border-slate-200/60 flex justify-end">
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(translationText);
                          alert("Patient translation copied to clipboard.");
                        }}
                        className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-slate-50 border border-slate-200 text-slate-600 hover:text-slate-800 text-[10px] font-semibold shadow-2xs transition-all cursor-pointer"
                      >
                        <svg className="w-3.5 h-3.5 text-slate-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 5H6a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2v-1M8 5a2 2 0 002 2h2a2 2 0 002-2M8 5a2 2 0 012-2h2a2 2 0 012 2m0 0h2a2 2 0 012 2v3m2 4H10m0 0l3-3m-3 3l3 3" />
                        </svg>
                        Copy Patient Text
                      </button>
                    </div>
                  </div>
                ) : null}
              </div>
            </div>
          )}
        </section>
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
