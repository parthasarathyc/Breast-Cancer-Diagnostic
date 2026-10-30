import React, { useState } from 'react';
import axios from 'axios';
import {
  Activity,
  FileDown,
  ShieldCheck,
  AlertCircle,
  Eye,
  CheckCircle2,
  Upload,
  Calendar,
  Sparkles,
  ChevronRight,
  Stethoscope,
  RefreshCw
} from 'lucide-react';

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [activePreset, setActivePreset] = useState(null);

  // Clinical inputs
  const [patientId] = useState('PT-2026-8841');
  const [patientAge, setPatientAge] = useState(48);
  const [familyHistory, setFamilyHistory] = useState(false);
  const [breastDensity, setBreastDensity] = useState('B - Scattered fibroglandular');
  const [symptoms] = useState('Routine screening');
  const [menopauseStatus, setMenopauseStatus] = useState('perimenopausal');

  // Analysis states
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [activeViewTab, setActiveViewTab] = useState('side_by_side'); // 'side_by_side' | 'gradcam' | 'segmentation'
  const [errorMsg, setErrorMsg] = useState(null);

  // Clinical Presets
  const loadPreset = async (presetType) => {
    setActivePreset(presetType);
    setSelectedFile(null);
    setLoading(true);
    setErrorMsg(null);

    try {
      const response = await axios.post(`/api/v1/analyze/sample/${presetType}`, null, {
        params: {
          patient_age: patientAge,
          family_history: familyHistory || presetType === 'malignant',
          breast_density: breastDensity
        }
      });
      setResults(response.data);
      if (response.data.agent1_detection?.overlay_detection_base64) {
        setPreviewUrl(response.data.agent1_detection.overlay_detection_base64);
      }
    } catch (err) {
      console.error(err);
      setErrorMsg(`Failed to process preset scan: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setActivePreset(null);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setErrorMsg(null);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setActivePreset(null);
      setPreviewUrl(URL.createObjectURL(file));
      setResults(null);
      setErrorMsg(null);
    }
  };

  const runAnalysis = async () => {
    if (!selectedFile && !activePreset) {
      setErrorMsg("Please upload an ultrasound image or select a patient study preset.");
      return;
    }

    if (activePreset) {
      return loadPreset(activePreset);
    }

    setLoading(true);
    setErrorMsg(null);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('patient_age', patientAge);
    formData.append('family_history', familyHistory);
    formData.append('breast_density', breastDensity);
    formData.append('symptoms', symptoms);
    formData.append('menopause_status', menopauseStatus);

    try {
      const response = await axios.post('/api/v1/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      setResults(response.data);
    } catch (err) {
      console.error(err);
      setErrorMsg(`Diagnostic pipeline failed: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };

  // BI-RADS badge styler
  const getBiradsTheme = (birads) => {
    if (!birads) return { badge: 'bg-emerald-50 text-emerald-800 border-emerald-200', dot: 'bg-emerald-500' };
    if (birads.includes('1') || birads.includes('2')) {
      return { badge: 'bg-emerald-50 text-emerald-800 border-emerald-200', dot: 'bg-emerald-500' };
    }
    if (birads.includes('3')) {
      return { badge: 'bg-amber-50 text-amber-800 border-amber-200', dot: 'bg-amber-500' };
    }
    return { badge: 'bg-rose-50 text-rose-800 border-rose-200', dot: 'bg-rose-500' };
  };

  return (
    <div className="min-h-screen bg-[#f8faf9] text-slate-800 font-sans antialiased selection:bg-emerald-100 selection:text-emerald-900 flex flex-col">
      {/* Top Clinical Header - Responsive for 320px+ */}
      <header className="bg-white border-b border-slate-200/80 sticky top-0 z-30 shadow-[0_1px_3px_rgba(0,0,0,0.02)]">
        <div className="max-w-7xl mx-auto px-3 sm:px-6 lg:px-8 h-14 sm:h-16 flex items-center justify-between">
          {/* Logo & Brand */}
          <div className="flex items-center space-x-2.5 sm:space-x-3.5 min-w-0">
            <div className="h-8 w-8 sm:h-10 sm:w-10 rounded-xl bg-gradient-to-tr from-emerald-700 to-emerald-500 text-white flex items-center justify-center shrink-0 shadow-xs shadow-emerald-700/20">
              <Activity className="h-4 w-4 sm:h-5 sm:w-5 stroke-[2.5]" />
            </div>
            <div className="min-w-0">
              <div className="flex items-center space-x-1.5 sm:space-x-2">
                <span className="text-sm sm:text-base font-bold tracking-tight text-slate-900 truncate">
                  Breast CARE-MA
                </span>
                <span className="text-[9px] sm:text-[11px] font-semibold uppercase tracking-wider bg-emerald-50 text-emerald-700 px-1.5 py-0.5 rounded border border-emerald-200/70 shrink-0">
                  CADx
                </span>
              </div>
              <p className="text-[10px] sm:text-[11px] text-slate-400 font-medium truncate hidden xs:block">
                Ultrasound Diagnostic Decision Support Station
              </p>
            </div>
          </div>

          {/* Right Status Indicators */}
          <div className="flex items-center space-x-2 sm:space-x-4 shrink-0">
            <div className="flex items-center space-x-1.5 bg-slate-50 border border-slate-200/80 px-2 py-1 rounded-lg text-[10px] sm:text-xs font-medium text-slate-600">
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse shrink-0"></span>
              <span className="hidden sm:inline">AI Engine Online</span>
              <span className="sm:hidden">Online</span>
            </div>

            <div className="hidden md:flex items-center space-x-1.5 text-xs text-slate-500 pl-2 border-l border-slate-200">
              <Calendar className="w-3.5 h-3.5 text-slate-400" />
              <span>{new Date().toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Main Workspace */}
      <main className="max-w-7xl mx-auto w-full px-3 sm:px-6 lg:px-8 py-4 sm:py-6 flex-1">
        {errorMsg && (
          <div className="mb-4 sm:mb-6 p-3 sm:p-3.5 bg-rose-50 border border-rose-200/80 rounded-xl flex items-center justify-between text-rose-800 text-xs font-medium">
            <div className="flex items-center space-x-2 shrink-0">
              <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
              <span className="break-all">{errorMsg}</span>
            </div>
            <button onClick={() => setErrorMsg(null)} className="text-rose-500 hover:text-rose-700 ml-2">Dismiss</button>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">
          {/* ======================================================== */}
          {/* LEFT COLUMN: Patient Intake & Scan Control (5 cols)      */}
          {/* ======================================================== */}
          <div className="lg:col-span-5 space-y-4 sm:space-y-5">
            {/* 1. Ultrasound Image Intake Card */}
            <div className="bg-white rounded-2xl border border-slate-200/80 p-3.5 sm:p-5 shadow-xs">
              <div className="flex items-center justify-between mb-2.5">
                <label className="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                  <Upload className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Scan Acquisition</span>
                </label>
              </div>

              {/* Upload Dropzone */}
              <div
                onDragOver={(e) => e.preventDefault()}
                onDrop={handleDrop}
                className={`relative border-2 border-dashed rounded-xl p-3 sm:p-5 text-center transition-all ${
                  previewUrl
                    ? 'border-emerald-400 bg-emerald-50/20'
                    : 'border-slate-200 hover:border-emerald-400 bg-slate-50/50 hover:bg-emerald-50/10'
                }`}
              >
                {previewUrl ? (
                  <div className="space-y-2 sm:space-y-3">
                    <div className="relative rounded-lg overflow-hidden bg-black aspect-video max-h-36 sm:max-h-40 mx-auto flex items-center justify-center">
                      <img src={previewUrl} alt="Patient Ultrasound" className="object-contain h-full w-full" />
                    </div>
                    <div className="flex items-center justify-between text-[10px] sm:text-[11px] text-slate-500 pt-0.5">
                      <span className="truncate max-w-[130px] sm:max-w-[200px] font-medium text-slate-700">
                        {selectedFile ? selectedFile.name : `${activePreset?.toUpperCase()} Study`}
                      </span>
                      <label className="text-emerald-700 font-semibold cursor-pointer hover:underline shrink-0 ml-2">
                        Change Scan
                        <input type="file" accept="image/*" onChange={handleFileChange} className="sr-only" />
                      </label>
                    </div>
                  </div>
                ) : (
                  <label className="cursor-pointer block py-1.5 sm:py-2">
                    <div className="h-9 w-9 sm:h-10 sm:w-10 mx-auto mb-1.5 rounded-full bg-emerald-50 text-emerald-600 flex items-center justify-center">
                      <Upload className="h-4 w-4 sm:h-5 sm:w-5" />
                    </div>
                    <span className="text-xs font-semibold text-slate-800 block">Click to upload DICOM / Image</span>
                    <span className="text-[10px] sm:text-[11px] text-slate-400 block mt-0.5">or drag and drop scan file</span>
                    <input type="file" accept="image/*" onChange={handleFileChange} className="sr-only" />
                  </label>
                )}
              </div>

              {/* Quick Case Presets */}
              <div className="mt-3 pt-2.5 border-t border-slate-100">
                <span className="text-[10px] sm:text-[11px] font-semibold text-slate-400 block mb-1.5">Reference Studies:</span>
                <div className="grid grid-cols-3 gap-1 sm:gap-1.5">
                  <button
                    type="button"
                    onClick={() => loadPreset('benign')}
                    disabled={loading}
                    className={`py-1.5 px-1 sm:px-2 rounded-lg text-[10px] sm:text-[11px] font-semibold border transition-all truncate ${
                      activePreset === 'benign'
                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-xs'
                        : 'bg-white hover:bg-emerald-50/70 text-slate-600 border-slate-200'
                    }`}
                  >
                    Benign
                  </button>
                  <button
                    type="button"
                    onClick={() => loadPreset('malignant')}
                    disabled={loading}
                    className={`py-1.5 px-1 sm:px-2 rounded-lg text-[10px] sm:text-[11px] font-semibold border transition-all truncate ${
                      activePreset === 'malignant'
                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-xs'
                        : 'bg-white hover:bg-emerald-50/70 text-slate-600 border-slate-200'
                    }`}
                  >
                    Malignant
                  </button>
                  <button
                    type="button"
                    onClick={() => loadPreset('normal')}
                    disabled={loading}
                    className={`py-1.5 px-1 sm:px-2 rounded-lg text-[10px] sm:text-[11px] font-semibold border transition-all truncate ${
                      activePreset === 'normal'
                        ? 'bg-emerald-600 text-white border-emerald-600 shadow-xs'
                        : 'bg-white hover:bg-emerald-50/70 text-slate-600 border-slate-200'
                    }`}
                  >
                    Clear Scan
                  </button>
                </div>
              </div>
            </div>

            {/* 2. Patient Clinical Context Card */}
            <div className="bg-white rounded-2xl border border-slate-200/80 p-3.5 sm:p-5 shadow-xs space-y-3.5 sm:space-y-4">
              <div className="flex items-center justify-between pb-2 border-b border-slate-100">
                <span className="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                  <Stethoscope className="w-3.5 h-3.5 text-emerald-600" />
                  <span>Biomarkers</span>
                </span>
                <span className="text-[10px] sm:text-[11px] font-mono text-slate-400 bg-slate-50 px-1.5 py-0.5 rounded border border-slate-200">
                  {patientId}
                </span>
              </div>

              {/* Age Slider */}
              <div className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-600 font-medium text-[11px] sm:text-xs">Patient Age</span>
                  <span className="font-bold text-emerald-700 text-xs sm:text-sm">{patientAge} yrs</span>
                </div>
                <input
                  type="range"
                  min="20"
                  max="85"
                  value={patientAge}
                  onChange={(e) => setPatientAge(parseInt(e.target.value))}
                  className="w-full accent-emerald-600 h-1.5 bg-slate-200 rounded-lg cursor-pointer"
                />
              </div>

              {/* Breast Density & Menopause - responsive flex/grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 sm:gap-3">
                <div className="space-y-1">
                  <label className="text-[10px] sm:text-[11px] font-medium text-slate-600 block">Menopause</label>
                  <select
                    value={menopauseStatus}
                    onChange={(e) => setMenopauseStatus(e.target.value)}
                    className="w-full px-2 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  >
                    <option value="premenopausal">Premenopausal</option>
                    <option value="perimenopausal">Perimenopausal</option>
                    <option value="postmenopausal">Postmenopausal</option>
                  </select>
                </div>

                <div className="space-y-1">
                  <label className="text-[10px] sm:text-[11px] font-medium text-slate-600 block">Breast Density</label>
                  <select
                    value={breastDensity}
                    onChange={(e) => setBreastDensity(e.target.value)}
                    className="w-full px-2 py-1.5 text-xs rounded-lg border border-slate-200 bg-white text-slate-800 focus:outline-none focus:ring-1 focus:ring-emerald-500"
                  >
                    <option value="A - Almost entirely fatty">Category A (Fatty)</option>
                    <option value="B - Scattered fibroglandular">Category B (Scattered)</option>
                    <option value="C - Heterogeneously dense">Category C (Dense)</option>
                    <option value="D - Extremely dense">Category D (Extremely)</option>
                  </select>
                </div>
              </div>

              {/* Family History Toggle */}
              <label className="flex items-center justify-between p-2 sm:p-2.5 bg-slate-50/70 hover:bg-slate-50 rounded-xl border border-slate-200/80 cursor-pointer transition-colors">
                <span className="text-[11px] sm:text-xs font-medium text-slate-700 pr-2">1st-Degree Family Malignancy</span>
                <input
                  type="checkbox"
                  checked={familyHistory}
                  onChange={(e) => setFamilyHistory(e.target.checked)}
                  className="h-4 w-4 rounded-sm border-slate-300 text-emerald-600 focus:ring-emerald-500 accent-emerald-600 shrink-0"
                />
              </label>

              {/* Primary Analyze Action Button */}
              <button
                type="button"
                onClick={runAnalysis}
                disabled={loading}
                className="w-full py-2.5 sm:py-3 px-3 sm:px-4 bg-emerald-600 hover:bg-emerald-700 active:bg-emerald-800 disabled:bg-slate-200 disabled:text-slate-400 text-white font-semibold text-xs tracking-wide rounded-xl shadow-xs transition-all flex items-center justify-center space-x-2"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-3.5 h-3.5 sm:w-4 sm:h-4 animate-spin text-white" />
                    <span>Processing Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Sparkles className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-200" />
                    <span>Compute Diagnostic Assessment</span>
                  </>
                )}
              </button>
            </div>
          </div>

          {/* ======================================================== */}
          {/* RIGHT COLUMN: Diagnostic Workstation & Findings (7 cols) */}
          {/* ======================================================== */}
          <div className="lg:col-span-7 space-y-4 sm:space-y-5">
            {!results ? (
              <div className="bg-white rounded-2xl border border-slate-200/80 p-8 sm:p-12 text-center text-slate-400 shadow-xs flex flex-col items-center justify-center min-h-[300px] sm:min-h-[500px]">
                <div className="h-12 w-12 sm:h-14 sm:w-14 rounded-2xl bg-emerald-50 text-emerald-600 flex items-center justify-center mb-3">
                  <Activity className="w-6 h-6 sm:w-7 sm:h-7 stroke-[1.8]" />
                </div>
                <h3 className="text-xs sm:text-sm font-bold text-slate-700">Diagnostic Station Ready</h3>
                <p className="text-[11px] sm:text-xs text-slate-400 max-w-xs sm:max-w-sm mt-1">
                  Upload an ultrasound scan or select a reference study on the left to compute the BI-RADS diagnostic assessment.
                </p>
              </div>
            ) : (
              <div className="space-y-4 sm:space-y-5">
                {/* 1. Primary Assessment Summary Banner */}
                <div className="bg-white rounded-2xl border border-slate-200/80 p-3.5 sm:p-5 shadow-xs">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between pb-3.5 border-b border-slate-100 gap-3">
                    <div className="min-w-0">
                      <div className="flex items-center space-x-1.5 sm:space-x-2 text-[10px] sm:text-[11px] font-medium text-slate-400 truncate">
                        <span>STUDY:</span>
                        <span className="font-mono font-bold text-slate-700 truncate">{results.case_id}</span>
                        <span>•</span>
                        <span>{results.pipeline_duration_seconds}s</span>
                      </div>
                      <div className="flex items-baseline space-x-2 mt-1">
                        <span className="text-[11px] uppercase tracking-wider text-slate-400 font-semibold">Diagnosis:</span>
                        <span className={`text-lg sm:text-xl font-extrabold tracking-tight capitalize ${
                          results.agent2_classification.predicted_class === 'malignant'
                            ? 'text-rose-700'
                            : (results.agent2_classification.predicted_class === 'benign' ? 'text-emerald-700' : 'text-slate-800')
                        }`}>
                          {results.agent2_classification.predicted_class}
                        </span>
                      </div>
                    </div>

                    <div className="flex items-center justify-between sm:justify-end space-x-2.5 sm:space-x-3 shrink-0">
                      {/* BI-RADS Category Pill */}
                      <div className={`px-2.5 sm:px-3.5 py-1.5 sm:py-2 rounded-xl border flex items-center space-x-2 ${getBiradsTheme(results.agent5_risk.birads_category).badge}`}>
                        <span className={`h-2 w-2 rounded-full ${getBiradsTheme(results.agent5_risk.birads_category).dot}`}></span>
                        <div>
                          <div className="text-xs sm:text-sm font-extrabold tracking-tight">{results.agent5_risk.birads_category}</div>
                          <div className="text-[9px] sm:text-[10px] font-semibold truncate max-w-[100px] sm:max-w-none">
                            {results.agent5_risk.risk_level}
                          </div>
                        </div>
                      </div>

                      {/* Download PDF CTA */}
                      <a
                        href={`/api/v1/reports/${results.case_id}/pdf`}
                        target="_blank"
                        rel="noreferrer"
                        className="px-2.5 sm:px-3.5 py-1.5 sm:py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-xl shadow-xs transition-colors flex items-center space-x-1.5 text-xs font-semibold shrink-0"
                      >
                        <FileDown className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
                        <span className="text-[11px] sm:text-xs">PDF</span>
                      </a>
                    </div>
                  </div>

                  {/* Quantitative Probabilities Bar - Responsive Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-3 pt-3 sm:pt-4">
                    <div className="bg-slate-50/70 p-2.5 sm:p-3 rounded-xl border border-slate-100">
                      <div className="flex justify-between items-center sm:block">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-0.5 sm:mb-1">
                          Malignancy Risk
                        </span>
                        <span className="text-sm sm:text-base font-bold text-slate-800">
                          {(results.agent5_risk.malignancy_risk_score * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1.5 sm:mt-2">
                        <div
                          className={`h-full ${
                            results.agent5_risk.malignancy_risk_score > 0.4 ? 'bg-rose-500' : (results.agent5_risk.malignancy_risk_score > 0.15 ? 'bg-amber-500' : 'bg-emerald-500')
                          }`}
                          style={{ width: `${Math.min(results.agent5_risk.malignancy_risk_score * 100, 100)}%` }}
                        />
                      </div>
                    </div>

                    <div className="bg-slate-50/70 p-2.5 sm:p-3 rounded-xl border border-slate-100">
                      <div className="flex justify-between items-center sm:block">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-0.5 sm:mb-1">
                          Benign Probability
                        </span>
                        <span className="text-sm sm:text-base font-bold text-slate-800">
                          {((results.agent2_classification.calibrated_probabilities.benign || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1.5 sm:mt-2">
                        <div
                          className="h-full bg-emerald-500"
                          style={{ width: `${((results.agent2_classification.calibrated_probabilities.benign || 0) * 100)}%` }}
                        />
                      </div>
                    </div>

                    <div className="bg-slate-50/70 p-2.5 sm:p-3 rounded-xl border border-slate-100">
                      <div className="flex justify-between items-center sm:block">
                        <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block mb-0.5 sm:mb-1">
                          Non-Lesional Normal
                        </span>
                        <span className="text-sm sm:text-base font-bold text-slate-800">
                          {((results.agent2_classification.calibrated_probabilities.normal || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="w-full bg-slate-200 h-1.5 rounded-full overflow-hidden mt-1.5 sm:mt-2">
                        <div
                          className="h-full bg-slate-400"
                          style={{ width: `${((results.agent2_classification.calibrated_probabilities.normal || 0) * 100)}%` }}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                {/* 2. Medical Image Viewer (PACS Grade) */}
                <div className="bg-white rounded-2xl border border-slate-200/80 p-3.5 sm:p-5 shadow-xs">
                  <div className="flex flex-col xs:flex-row items-start xs:items-center justify-between gap-2 mb-3">
                    <span className="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5">
                      <Eye className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                      <span>Imaging & Explainability</span>
                    </span>

                    {/* View Switcher - horizontally scrollable on mobile */}
                    <div className="inline-flex max-w-full overflow-x-auto bg-slate-100 p-0.5 rounded-lg text-[10px] sm:text-xs font-medium text-slate-600 shrink-0">
                      <button
                        onClick={() => setActiveViewTab('side_by_side')}
                        className={`px-2 py-1 rounded-md transition-all whitespace-nowrap ${
                          activeViewTab === 'side_by_side' ? 'bg-white shadow-xs text-emerald-800 font-semibold' : 'hover:text-slate-900'
                        }`}
                      >
                        Side-by-Side
                      </button>
                      <button
                        onClick={() => setActiveViewTab('gradcam')}
                        className={`px-2 py-1 rounded-md transition-all whitespace-nowrap ${
                          activeViewTab === 'gradcam' ? 'bg-white shadow-xs text-emerald-800 font-semibold' : 'hover:text-slate-900'
                        }`}
                      >
                        Heatmap
                      </button>
                      <button
                        onClick={() => setActiveViewTab('segmentation')}
                        className={`px-2 py-1 rounded-md transition-all whitespace-nowrap ${
                          activeViewTab === 'segmentation' ? 'bg-white shadow-xs text-emerald-800 font-semibold' : 'hover:text-slate-900'
                        }`}
                      >
                        Mask
                      </button>
                    </div>
                  </div>

                  {/* Image Display Grid */}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 sm:gap-4">
                    {/* Panel 1: Original / Localization */}
                    <div className="space-y-1">
                      <div className="flex justify-between items-center text-[10px] sm:text-[11px] font-semibold text-slate-500">
                        <span>LOCALIZED TUMOR BOUNDS</span>
                        <span className="text-emerald-700 font-mono">
                          {results.agent1_detection.lesion_area_percentage}% Area
                        </span>
                      </div>
                      <div className="bg-slate-950 rounded-xl overflow-hidden aspect-[4/3] flex items-center justify-center border border-slate-800 relative">
                        {results.agent1_detection.overlay_detection_base64 && (
                          <img
                            src={results.agent1_detection.overlay_detection_base64}
                            alt="Lesion Localization"
                            className="w-full h-full object-contain"
                          />
                        )}
                        <span className="absolute bottom-1.5 left-1.5 text-[9px] font-mono text-emerald-400 bg-black/60 px-1 py-0.5 rounded">
                          ROI
                        </span>
                      </div>
                    </div>

                    {/* Panel 2: XAI or Mask */}
                    <div className="space-y-1">
                      <div className="flex justify-between items-center text-[10px] sm:text-[11px] font-semibold text-slate-500">
                        <span className="truncate">
                          {activeViewTab === 'segmentation' ? 'SEGMENTATION MASK' : 'GRAD-CAM++ HEATMAP'}
                        </span>
                        <span className="text-slate-400 font-mono text-[10px]">XAI</span>
                      </div>
                      <div className="bg-slate-950 rounded-xl overflow-hidden aspect-[4/3] flex items-center justify-center border border-slate-800 relative">
                        {activeViewTab === 'segmentation' ? (
                          results.agent1_detection.mask_base64 && (
                            <img
                              src={results.agent1_detection.mask_base64}
                              alt="Binary Mask"
                              className="w-full h-full object-contain"
                            />
                          )
                        ) : (
                          results.agent3_xai.gradcam_overlay_base64 && (
                            <img
                              src={results.agent3_xai.gradcam_overlay_base64}
                              alt="Grad-CAM++ Overlay"
                              className="w-full h-full object-contain"
                            />
                          )
                        )}
                        <span className="absolute bottom-1.5 left-1.5 text-[9px] font-mono text-amber-400 bg-black/60 px-1 py-0.5 rounded">
                          {activeViewTab === 'segmentation' ? 'MASK' : 'ATTENTION'}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Morphological Descriptors Table - responsive wrapping */}
                  <div className="mt-3 pt-2.5 border-t border-slate-100 grid grid-cols-2 sm:grid-cols-5 gap-1.5 sm:gap-2 text-xs">
                    <div className="p-2 bg-slate-50/80 rounded-lg border border-slate-100">
                      <span className="text-[9px] sm:text-[10px] font-medium text-slate-400 block uppercase">Shape</span>
                      <span className="font-bold text-slate-800 capitalize text-[11px] sm:text-xs truncate block">
                        {results.agent3_xai.morphology.shape}
                      </span>
                    </div>
                    <div className="p-2 bg-slate-50/80 rounded-lg border border-slate-100">
                      <span className="text-[9px] sm:text-[10px] font-medium text-slate-400 block uppercase">Margin</span>
                      <span className="font-bold text-slate-800 capitalize text-[11px] sm:text-xs truncate block">
                        {results.agent3_xai.morphology.margin}
                      </span>
                    </div>
                    <div className="p-2 bg-slate-50/80 rounded-lg border border-slate-100">
                      <span className="text-[9px] sm:text-[10px] font-medium text-slate-400 block uppercase">Orientation</span>
                      <span className="font-bold text-slate-800 capitalize text-[11px] sm:text-xs truncate block">
                        {results.agent3_xai.morphology.orientation}
                      </span>
                    </div>
                    <div className="p-2 bg-slate-50/80 rounded-lg border border-slate-100">
                      <span className="text-[9px] sm:text-[10px] font-medium text-slate-400 block uppercase">Aspect Ratio</span>
                      <span className="font-bold text-slate-800 text-[11px] sm:text-xs">
                        {results.agent3_xai.morphology.aspect_ratio}
                      </span>
                    </div>
                    <div className="p-2 bg-slate-50/80 rounded-lg border border-slate-100 col-span-2 sm:col-span-1">
                      <span className="text-[9px] sm:text-[10px] font-medium text-slate-400 block uppercase">Acoustic Shadow</span>
                      <span className={`font-bold text-[11px] sm:text-xs ${results.agent3_xai.morphology.acoustic_shadowing_observed ? 'text-rose-600' : 'text-emerald-700'}`}>
                        {results.agent3_xai.morphology.acoustic_shadowing_observed ? 'Present' : 'Absent'}
                      </span>
                    </div>
                  </div>
                </div>

                {/* 3. Clinical Action Plan & Evidence Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4">
                  {/* Action Plan */}
                  <div className="bg-white rounded-2xl border border-slate-200/80 p-3.5 sm:p-5 shadow-xs flex flex-col justify-between">
                    <div>
                      <span className="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5 mb-2.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>Recommended Clinical Action</span>
                      </span>
                      <ul className="space-y-1.5 sm:space-y-2 text-xs text-slate-700">
                        {results.agent6_report.recommended_action.map((act, i) => (
                          <li key={i} className="flex items-start space-x-2">
                            <ChevronRight className="w-3.5 h-3.5 text-emerald-600 shrink-0 mt-0.5" />
                            <span className="text-[11px] sm:text-xs">{act}</span>
                          </li>
                        ))}
                      </ul>
                    </div>

                    <div className="mt-3 pt-2.5 border-t border-slate-100 text-[10px] sm:text-[11px] text-slate-400">
                      Protocol aligned with ACR BI-RADS 5th Edition follow-up pathway.
                    </div>
                  </div>

                  {/* Evidence Citation */}
                  <div className="bg-white rounded-2xl border border-slate-200/80 p-3.5 sm:p-5 shadow-xs flex flex-col justify-between">
                    <div>
                      <span className="text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-500 flex items-center space-x-1.5 mb-2.5">
                        <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                        <span>ACR BI-RADS Evidence Criteria</span>
                      </span>
                      <div className="space-y-1.5 sm:space-y-2 text-xs">
                        {results.agent4_evidence.matched_guidelines.slice(0, 2).map((g, idx) => (
                          <div key={idx} className="bg-slate-50/70 p-2 sm:p-2.5 rounded-lg border border-slate-100">
                            <div className="font-semibold text-slate-800 text-[10px] sm:text-[11px]">{g.title}</div>
                            <p className="text-slate-500 text-[10px] sm:text-[11px] mt-0.5 line-clamp-2">{g.guideline}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="mt-3 pt-2.5 border-t border-slate-100 text-[10px] sm:text-[11px] text-slate-400 flex items-center justify-between">
                      <span>Source: ACR Atlas</span>
                      <span className="font-semibold text-emerald-700">Verified</span>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </main>

      {/* Minimal Footer */}
      <footer className="mt-auto py-4 border-t border-slate-200/80 text-center text-[10px] sm:text-xs text-slate-400 px-4">
        <p>Breast CARE-MA Diagnostic Decision Support System • For Clinical Research & Evaluation</p>
      </footer>
    </div>
  );
}
