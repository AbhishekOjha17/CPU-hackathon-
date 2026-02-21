// app/customer/page.tsx
'use client'
import { useState, useCallback, useRef, ChangeEvent, DragEvent, useEffect } from 'react';

// Define types
interface WhatIfScenario {
  id: string;
  title: string;
  description?: string;
  impact: string;
  improvement_percentage: number;
  current_value: string;
  suggested_value: string;
  reasoning: string;
  action_plan?: string;
  parameter?: string;
}

interface RiskAnalysis {
  professional_risk: {
    category: string;
    industry_health: string;
    income_stability_score: number;
    specific_concerns: string[];
    future_outlook: string;
  };
  income_trajectory_analysis: {
    trend: string;
    stress_scenarios: Array<{
      scenario: string;
      impact: string;
      risk_level: string;
      survival_time_months?: string;
    }>;
  };
  macroeconomic_sensitivity: {
    recession_impact: string;
    industry_downturn_sensitivity: string;
    interest_rate_hike_impact: string;
    most_vulnerable_to: string[];
  };
  recommendations: {
    risk_mitigation: string[];
    monitoring_focus: string[];
  };
}

interface UploadResponse {
  success: boolean;
  message: string;
  prediction: {
    risk_level: string;
    probability: number;
  };
  what_if_scenarios: {
    combined_recommendations: WhatIfScenario[];
  };
  risk_analysis: RiskAnalysis;
  application_id: string;
  user_features_used?: boolean;
}

interface UserFeatures {
  monthly_income: number;
  employment_type: string;
  years_in_current_job: number;
  existing_emi_obligations: number;
  loan_amount_requested: number;
  loan_purpose: string;
  credit_score: number;
  marital_status: string;
  dependents_count: number;
  education_level: string;
  residence_type: string;
  age: number;
  co_applicant_income: number;
  property_type: string;
  loan_term: number;
}

export default function CustomerPage() {
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadProgress, setUploadProgress] = useState<number>(0);
  const [result, setResult] = useState<UploadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showForm, setShowForm] = useState<boolean>(false);
  const [userFeatures, setUserFeatures] = useState<Partial<UserFeatures>>({});
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Employment type options
  const employmentTypes = ['Salaried', 'Self-Employed', 'Business Owner', 'Freelancer', 'Retired', 'Unemployed'];
  const loanPurposes = ['Home Loan', 'Car Loan', 'Personal Loan', 'Education Loan', 'Business Loan', 'Debt Consolidation'];
  const maritalStatuses = ['Single', 'Married', 'Divorced', 'Widowed'];
  const educationLevels = ['High School', 'Graduate', 'Post Graduate', 'Doctorate', 'Professional'];
  const residenceTypes = ['Owned', 'Rented', 'Mortgaged', 'Living with Family'];
  const propertyTypes = ['Apartment', 'Independent House', 'Villa', 'Commercial', 'Land'];

  const handleDragEnter = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDrop = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const droppedFiles = Array.from(e.dataTransfer.files);
    setFiles(prev => [...prev, ...droppedFiles]);
  }, []);

  const handleFileSelect = useCallback((e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files);
      setFiles(prev => [...prev, ...selectedFiles]);
    }
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles(prev => prev.filter((_, i) => i !== index));
  }, []);

  const handleInputChange = (field: keyof UserFeatures, value: string | number) => {
    setUserFeatures(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleUpload = async () => {
    if (files.length === 0) {
      setError('Please select at least one file');
      return;
    }

    setIsUploading(true);
    setError(null);
    setResult(null);
    setUploadProgress(0);

    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });

    // Add user features if any are filled
    const hasFeatures = Object.keys(userFeatures).length > 0;
    if (hasFeatures) {
      formData.append('user_features', JSON.stringify(userFeatures));
    }

    const progressInterval = setInterval(() => {
      setUploadProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const response = await fetch('http://localhost:8000/api/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(progressInterval);
      setUploadProgress(100);

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.statusText}`);
      }

      const data: UploadResponse = await response.json();
      setResult(data);
      setFiles([]);
      
      setTimeout(() => setUploadProgress(0), 1000);
    } catch (err) {
      clearInterval(progressInterval);
      setError(err instanceof Error ? err.message : 'An unknown error occurred');
      setUploadProgress(0);
    } finally {
      setIsUploading(false);
    }
  };

  const getRiskLevelText = (riskLevel: string): { title: string; description: string; color: string } => {
    switch (riskLevel?.toLowerCase()) {
      case 'high risk':
        return {
          title: 'Loan Application Not Approved',
          description: 'Based on your documents, your application has a high risk profile.',
          color: 'text-red-700 bg-red-50 border-red-200'
        };
      case 'medium risk':
        return {
          title: 'Loan Application Under Review',
          description: 'Your application shows some risk factors that need consideration.',
          color: 'text-yellow-700 bg-yellow-50 border-yellow-200'
        };
      case 'low risk':
        return {
          title: 'Loan Application Approved',
          description: 'Congratulations! Your application has a low risk profile.',
          color: 'text-green-700 bg-green-50 border-green-200'
        };
      default:
        return {
          title: 'Application Processed',
          description: 'Your documents have been analyzed.',
          color: 'text-gray-700 bg-gray-50 border-gray-200'
        };
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="min-h-screen bg-white">
      {/* Simple Header */}
      <div className="border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-2xl font-bold text-gray-900">Loan Application Assistant</h1>
          <p className="text-sm text-gray-500">Upload your documents to check loan eligibility</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Upload Section */}
          <div className="space-y-6">
            {/* Toggle Form Button */}
            <button
              onClick={() => setShowForm(!showForm)}
              className="text-sm text-gray-600 hover:text-gray-900 flex items-center"
            >
              <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6v6m0 0v6m0-6h6m-6 0H6" />
              </svg>
              {showForm ? 'Hide' : 'Show'} additional details form
            </button>

            {/* User Features Form */}
            {showForm && (
              <div className="border border-gray-200 rounded-lg p-6 bg-gray-50">
                <h3 className="text-md font-medium text-gray-900 mb-4">Tell us more about yourself</h3>
                <p className="text-xs text-gray-500 mb-4">This helps us provide more accurate assessment</p>
                
                <div className="space-y-4 max-h-96 overflow-y-auto pr-2">
                  {/* Monthly Income */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Monthly Income (₹)</label>
                    <input
                      type="number"
                      value={userFeatures.monthly_income || ''}
                      onChange={(e) => handleInputChange('monthly_income', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 50000"
                    />
                  </div>

                  {/* Employment Type */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Employment Type</label>
                    <select
                      value={userFeatures.employment_type || ''}
                      onChange={(e) => handleInputChange('employment_type', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    >
                      <option value="">Select</option>
                      {employmentTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </div>

                  {/* Years in Current Job */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Years in Current Job</label>
                    <input
                      type="number"
                      value={userFeatures.years_in_current_job || ''}
                      onChange={(e) => handleInputChange('years_in_current_job', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 3"
                      step="0.5"
                    />
                  </div>

                  {/* Existing EMI Obligations */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Existing EMI Obligations (₹/month)</label>
                    <input
                      type="number"
                      value={userFeatures.existing_emi_obligations || ''}
                      onChange={(e) => handleInputChange('existing_emi_obligations', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 10000"
                    />
                  </div>

                  {/* Loan Amount Requested */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Loan Amount Requested (₹)</label>
                    <input
                      type="number"
                      value={userFeatures.loan_amount_requested || ''}
                      onChange={(e) => handleInputChange('loan_amount_requested', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 500000"
                    />
                  </div>

                  {/* Loan Purpose */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Loan Purpose</label>
                    <select
                      value={userFeatures.loan_purpose || ''}
                      onChange={(e) => handleInputChange('loan_purpose', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    >
                      <option value="">Select</option>
                      {loanPurposes.map(purpose => (
                        <option key={purpose} value={purpose}>{purpose}</option>
                      ))}
                    </select>
                  </div>

                  {/* Credit Score */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Credit Score (CIBIL)</label>
                    <input
                      type="number"
                      value={userFeatures.credit_score || ''}
                      onChange={(e) => handleInputChange('credit_score', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 750"
                      min="300"
                      max="900"
                    />
                  </div>

                  {/* Marital Status */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Marital Status</label>
                    <select
                      value={userFeatures.marital_status || ''}
                      onChange={(e) => handleInputChange('marital_status', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    >
                      <option value="">Select</option>
                      {maritalStatuses.map(status => (
                        <option key={status} value={status}>{status}</option>
                      ))}
                    </select>
                  </div>

                  {/* Dependents Count */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Number of Dependents</label>
                    <input
                      type="number"
                      value={userFeatures.dependents_count || ''}
                      onChange={(e) => handleInputChange('dependents_count', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 2"
                      min="0"
                    />
                  </div>

                  {/* Education Level */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Education Level</label>
                    <select
                      value={userFeatures.education_level || ''}
                      onChange={(e) => handleInputChange('education_level', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    >
                      <option value="">Select</option>
                      {educationLevels.map(level => (
                        <option key={level} value={level}>{level}</option>
                      ))}
                    </select>
                  </div>

                  {/* Residence Type */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Residence Type</label>
                    <select
                      value={userFeatures.residence_type || ''}
                      onChange={(e) => handleInputChange('residence_type', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    >
                      <option value="">Select</option>
                      {residenceTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </div>

                  {/* Age */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Age</label>
                    <input
                      type="number"
                      value={userFeatures.age || ''}
                      onChange={(e) => handleInputChange('age', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 35"
                      min="18"
                      max="100"
                    />
                  </div>

                  {/* Co-applicant Income */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Co-applicant Income (₹/month)</label>
                    <input
                      type="number"
                      value={userFeatures.co_applicant_income || ''}
                      onChange={(e) => handleInputChange('co_applicant_income', parseFloat(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 25000"
                    />
                  </div>

                  {/* Property Type */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Property Type</label>
                    <select
                      value={userFeatures.property_type || ''}
                      onChange={(e) => handleInputChange('property_type', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                    >
                      <option value="">Select</option>
                      {propertyTypes.map(type => (
                        <option key={type} value={type}>{type}</option>
                      ))}
                    </select>
                  </div>

                  {/* Loan Term */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Loan Term (months)</label>
                    <input
                      type="number"
                      value={userFeatures.loan_term || ''}
                      onChange={(e) => handleInputChange('loan_term', parseInt(e.target.value))}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm"
                      placeholder="e.g., 240"
                      min="12"
                      max="360"
                    />
                  </div>
                </div>
              </div>
            )}

            {/* Upload Area */}
            <div
              className={`border-2 border-dashed rounded-lg p-8 ${
                isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 bg-white'
              }`}
              onDragEnter={handleDragEnter}
              onDragLeave={handleDragLeave}
              onDragOver={handleDragOver}
              onDrop={handleDrop}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                onChange={handleFileSelect}
                className="hidden"
                accept=".txt,.pdf,.doc,.docx,.csv"
              />
              
              <div className="text-center">
                <div className="flex justify-center mb-4">
                  <div className="w-16 h-16 bg-gray-900 rounded-lg flex items-center justify-center">
                    <svg className="w-8 h-8 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                    </svg>
                  </div>
                </div>
                <h3 className="text-lg font-medium text-gray-900 mb-2">
                  Drag & drop your documents
                </h3>
                <p className="text-sm text-gray-500 mb-4">
                  or click to browse
                </p>
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="px-4 py-2 bg-gray-900 text-white text-sm font-medium rounded hover:bg-gray-800"
                >
                  Select Files
                </button>
                <p className="text-xs text-gray-400 mt-4">
                  PDF, TXT, DOC, DOCX, CSV (Max: 10MB per file)
                </p>
              </div>
            </div>

            {/* File List */}
            {files.length > 0 && (
              <div className="border border-gray-200 rounded-lg">
                <div className="px-4 py-3 bg-gray-50 border-b border-gray-200">
                  <p className="text-sm font-medium text-gray-700">Selected Files ({files.length})</p>
                </div>
                <div className="divide-y divide-gray-200">
                  {files.map((file, index) => (
                    <div key={index} className="px-4 py-3 flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <div>
                          <p className="text-sm text-gray-900">{file.name}</p>
                          <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                        </div>
                      </div>
                      <button
                        onClick={() => removeFile(index)}
                        className="text-gray-400 hover:text-red-500"
                      >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Upload Button */}
            {files.length > 0 && (
              <div className="border border-gray-200 rounded-lg p-4">
                <button
                  onClick={handleUpload}
                  disabled={isUploading}
                  className={`w-full py-3 px-4 rounded font-medium text-white ${
                    isUploading ? 'bg-gray-400 cursor-not-allowed' : 'bg-gray-900 hover:bg-gray-800'
                  }`}
                >
                  {isUploading ? (
                    <div className="flex items-center justify-center">
                      <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                      </svg>
                      Processing... {uploadProgress}%
                    </div>
                  ) : (
                    'Check Eligibility'
                  )}
                </button>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Customer-Friendly Results */}
          <div className="space-y-6">
            {result ? (
              <>
                {/* Simple Result Card */}
                <div className="border border-gray-200 rounded-lg">
                  <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                    <h2 className="text-lg font-medium text-gray-900">Application Status</h2>
                    {result.user_features_used && (
                      <span className="text-xs text-green-600 mt-1 block">✓ Using your provided information</span>
                    )}
                  </div>
                  
                  <div className="p-6">
                    {/* Risk Level Message */}
                    <div className={`p-4 rounded-lg border ${getRiskLevelText(result.prediction?.risk_level).color} mb-6`}>
                      <h3 className="text-lg font-medium mb-1">{getRiskLevelText(result.prediction?.risk_level).title}</h3>
                      <p className="text-sm">{getRiskLevelText(result.prediction?.risk_level).description}</p>
                    </div>

                    {/* Why Not Approved Section - Only show for High Risk */}
                    {result.prediction?.risk_level?.toLowerCase() === 'high risk' && (
                      <div className="mb-6">
                        <h3 className="text-md font-medium text-gray-900 mb-3">Why your application wasn't approved:</h3>
                        
                        {/* Professional Risk Concerns */}
                        {result.risk_analysis?.professional_risk?.specific_concerns?.length > 0 && (
                          <div className="mb-4">
                            <p className="text-sm font-medium text-gray-700 mb-2">Employment Related:</p>
                            <ul className="space-y-2">
                              {result.risk_analysis.professional_risk.specific_concerns.map((concern, idx) => (
                                <li key={idx} className="flex items-start text-sm text-gray-600">
                                  <span className="text-red-500 mr-2">•</span>
                                  {concern}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}

                        {/* Income Stability */}
                        {result.risk_analysis?.income_trajectory_analysis?.trend && (
                          <div className="mb-4">
                            <p className="text-sm font-medium text-gray-700 mb-2">Income Stability:</p>
                            <p className="text-sm text-gray-600">
                              Your income shows a {result.risk_analysis.income_trajectory_analysis.trend} trend, which concerns lenders.
                            </p>
                          </div>
                        )}

                        {/* Macroeconomic Vulnerabilities */}
                        {result.risk_analysis?.macroeconomic_sensitivity?.most_vulnerable_to?.length > 0 && (
                          <div className="mb-4">
                            <p className="text-sm font-medium text-gray-700 mb-2">Risk Factors:</p>
                            <ul className="space-y-2">
                              {result.risk_analysis.macroeconomic_sensitivity.most_vulnerable_to.map((factor, idx) => (
                                <li key={idx} className="flex items-start text-sm text-gray-600">
                                  <span className="text-red-500 mr-2">•</span>
                                  {factor}
                                </li>
                              ))}
                            </ul>
                          </div>
                        )}
                      </div>
                    )}

                    {/* What You Can Improve Section */}
                    {result.what_if_scenarios?.combined_recommendations?.length > 0 && (
                      <div className="border-t border-gray-200 pt-6">
                        <h3 className="text-md font-medium text-gray-900 mb-4">How to improve your chances:</h3>
                        <div className="space-y-4">
                          {result.what_if_scenarios.combined_recommendations.slice(0, 3).map((scenario) => (
                            <div key={scenario.id} className="border border-gray-200 rounded-lg p-4">
                              <h4 className="text-sm font-medium text-gray-900 mb-2">{scenario.title}</h4>
                              <p className="text-sm text-gray-600 mb-2">{scenario.reasoning}</p>
                              {scenario.action_plan && (
                                <div className="mt-2 pt-2 border-t border-gray-100">
                                  <p className="text-xs text-gray-500">
                                    <span className="font-medium">Suggestion:</span> {scenario.action_plan}
                                  </p>
                                </div>
                              )}
                              {scenario.improvement_percentage && (
                                <div className="mt-2">
                                  <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded">
                                    Could improve chances by {scenario.improvement_percentage}%
                                  </span>
                                </div>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* Application ID */}
                    <div className="mt-6 pt-4 border-t border-gray-200">
                      <p className="text-xs text-gray-400">
                        Application ID: {result.application_id}
                      </p>
                    </div>
                  </div>
                </div>
              </>
            ) : (
              // Empty State
              <div className="border border-gray-200 rounded-lg p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-lg flex items-center justify-center">
                  <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
                <h3 className="text-md font-medium text-gray-900 mb-2">No Application Yet</h3>
                <p className="text-sm text-gray-500">Upload your documents to check eligibility</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}