'use client'
import { useState, useEffect } from 'react';

interface Application {
  application_id: string;
  created_at: string;
  files_processed: string[];
  prediction_result: {
    risk_level: string;
    probability: number;
    confidence: number;
  };
  extracted_features: Record<string, any>;
  what_if_scenarios: {
    combined_recommendations: Array<{
      id: string;
      title: string;
      description: string;
      impact: string;
      improvement_percentage: number;
      current_value: string;
      suggested_value: string;
      reasoning: string;
    }>;
  };
  risk_analysis: {
    professional_risk: {
      category: string;
      industry_health: string;
      income_stability_score: number;
      specific_concerns: string[];
    };
    income_trajectory_analysis: {
      trend: string;
      stress_scenarios: Array<{
        scenario: string;
        impact: string;
        risk_level: string;
      }>;
    };
    macroeconomic_sensitivity: {
      recession_impact: string;
      industry_downturn_sensitivity: string;
      interest_rate_hike_impact: string;
      most_vulnerable_to: string[];
    };
  };
  features_count: number;
}

export default function AdminPage() {
  const [applications, setApplications] = useState<Application[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedApp, setSelectedApp] = useState<Application | null>(null);
  const [showDetailModal, setShowDetailModal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [stats, setStats] = useState({
    total: 0,
    highRisk: 0,
    mediumRisk: 0,
    lowRisk: 0
  });

  useEffect(() => {
    fetchApplications();
  }, []);

  const fetchApplications = async () => {
    try {
      setLoading(true);
      const response = await fetch('http://localhost:8000/api/applications?limit=100');
      const data = await response.json();
      
      if (data.success) {
        setApplications(data.applications);
        
        // Calculate stats
        const highRisk = data.applications.filter((app: Application) => 
          app.prediction_result?.risk_level?.toLowerCase().includes('high')).length;
        const mediumRisk = data.applications.filter((app: Application) => 
          app.prediction_result?.risk_level?.toLowerCase().includes('medium')).length;
        const lowRisk = data.applications.filter((app: Application) => 
          app.prediction_result?.risk_level?.toLowerCase().includes('low')).length;
        
        setStats({
          total: data.applications.length,
          highRisk,
          mediumRisk,
          lowRisk
        });
      } else {
        setError('Failed to fetch applications');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch applications');
    } finally {
      setLoading(false);
    }
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel?.toLowerCase()) {
      case 'high risk':
        return 'text-red-700 bg-red-50 border-red-200';
      case 'medium risk':
        return 'text-yellow-700 bg-yellow-50 border-yellow-200';
      case 'low risk':
        return 'text-green-700 bg-green-50 border-green-200';
      default:
        return 'text-gray-700 bg-gray-50 border-gray-200';
    }
  };

  // Feature importance for visualization (simplified SHAP-like values)
  const getFeatureImportance = (features: Record<string, any>) => {
    const importantFeatures = [
      { name: 'Credit Score', value: features.credit_score || 650, importance: 0.25 },
      { name: 'DTI Ratio', value: features.debt_to_income_ratio || 0.3, importance: 0.20 },
      { name: 'Income', value: features.monthly_income || 50000, importance: 0.15 },
      { name: 'Loan Amount', value: features.loan_amount_requested || 500000, importance: 0.12 },
      { name: 'Employment Years', value: features.years_in_current_job || 2, importance: 0.10 },
      { name: 'Past Defaults', value: features.past_defaults_count || 0, importance: 0.08 },
      { name: 'Age', value: features.applicant_age || 35, importance: 0.05 },
      { name: 'Existing EMI', value: features.existing_emi_obligations || 0, importance: 0.05 }
    ];
    return importantFeatures;
  };

  const DetailModal = ({ application, onClose }: { application: Application; onClose: () => void }) => {
    const features = application.extracted_features || {};
    const importance = getFeatureImportance(features);
    const prediction = application.prediction_result || {};

    return (
      <div className="fixed inset-0 bg-black bg-opacity-50 z-50 overflow-y-auto">
        <div className="min-h-screen px-4 py-8">
          <div className="bg-white max-w-6xl mx-auto rounded-lg border border-gray-200">
            {/* Modal Header */}
            <div className="px-6 py-4 border-b border-gray-200 flex justify-between items-center bg-gray-50">
              <h2 className="text-lg font-medium text-gray-900">Application Details</h2>
              <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
                <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Application ID and Date */}
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="text-gray-500">Application ID:</span>
                  <span className="ml-2 font-mono text-gray-900">{application.application_id}</span>
                </div>
                <div>
                  <span className="text-gray-500">Created:</span>
                  <span className="ml-2 text-gray-900">{formatDate(application.created_at)}</span>
                </div>
              </div>

              {/* Risk Level Card */}
              <div className={`p-4 rounded-lg border ${getRiskColor(prediction.risk_level)}`}>
                <div className="flex justify-between items-center">
                  <div>
                    <h3 className="text-lg font-medium">Risk Assessment</h3>
                    <p className="text-sm mt-1">Confidence: {(prediction.confidence * 100).toFixed(1)}%</p>
                  </div>
                  <div className="text-right">
                    <div className="text-2xl font-bold">{prediction.risk_level || 'Unknown'}</div>
                    <div className="text-sm">Probability: {(prediction.probability * 100).toFixed(1)}%</div>
                  </div>
                </div>
              </div>

              {/* Feature Importance Visualization */}
              <div className="border border-gray-200 rounded-lg p-4">
                <h3 className="text-md font-medium text-gray-900 mb-4">Feature Impact Analysis</h3>
                <div className="space-y-3">
                  {importance.map((feat, idx) => (
                    <div key={idx}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">{feat.name}</span>
                        <span className="text-gray-900 font-medium">
                          {typeof feat.value === 'number' ? feat.value.toLocaleString() : feat.value}
                        </span>
                      </div>
                      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-gray-900 rounded-full"
                          style={{ width: `${feat.importance * 100}%` }}
                        />
                      </div>
                      <div className="text-xs text-gray-500 mt-1">
                        Impact: {(feat.importance * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Two Column Layout for Detailed Analysis */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Professional Risk */}
                {application.risk_analysis?.professional_risk && (
                  <div className="border border-gray-200 rounded-lg p-4">
                    <h3 className="text-md font-medium text-gray-900 mb-3">Professional Risk</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Category:</span>
                        <span className="text-gray-900">{application.risk_analysis.professional_risk.category}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Industry Health:</span>
                        <span className="text-gray-900">{application.risk_analysis.professional_risk.industry_health}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Income Stability:</span>
                        <span className="text-gray-900">{application.risk_analysis.professional_risk.income_stability_score}/100</span>
                      </div>
                      <div className="mt-2">
                        <span className="text-gray-500">Concerns:</span>
                        <ul className="mt-1 space-y-1">
                          {application.risk_analysis.professional_risk.specific_concerns.map((concern, idx) => (
                            <li key={idx} className="text-sm text-gray-600 flex items-start">
                              <span className="text-red-500 mr-2">•</span>
                              {concern}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}

                {/* Income Trajectory */}
                {application.risk_analysis?.income_trajectory_analysis && (
                  <div className="border border-gray-200 rounded-lg p-4">
                    <h3 className="text-md font-medium text-gray-900 mb-3">Income Trajectory</h3>
                    <div className="space-y-3">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Trend:</span>
                        <span className="text-gray-900">{application.risk_analysis.income_trajectory_analysis.trend}</span>
                      </div>
                      <div>
                        <span className="text-gray-500">Stress Scenarios:</span>
                        <div className="mt-2 space-y-2">
                          {application.risk_analysis.income_trajectory_analysis.stress_scenarios.map((scenario, idx) => (
                            <div key={idx} className="text-xs p-2 bg-gray-50 rounded">
                              <div className="font-medium text-gray-900">{scenario.scenario}</div>
                              <div className="text-gray-600 mt-1">{scenario.impact}</div>
                              <div className={`mt-1 inline-block px-2 py-0.5 rounded ${
                                scenario.risk_level === 'high' ? 'bg-red-50 text-red-700' :
                                scenario.risk_level === 'medium' ? 'bg-yellow-50 text-yellow-700' :
                                'bg-green-50 text-green-700'
                              }`}>
                                {scenario.risk_level} risk
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* Macroeconomic Sensitivity */}
                {application.risk_analysis?.macroeconomic_sensitivity && (
                  <div className="border border-gray-200 rounded-lg p-4">
                    <h3 className="text-md font-medium text-gray-900 mb-3">Macroeconomic Sensitivity</h3>
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Recession Impact:</span>
                        <span className="text-gray-900">{application.risk_analysis.macroeconomic_sensitivity.recession_impact}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Industry Downturn:</span>
                        <span className="text-gray-900">{application.risk_analysis.macroeconomic_sensitivity.industry_downturn_sensitivity}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Interest Rate Impact:</span>
                        <span className="text-gray-900">{application.risk_analysis.macroeconomic_sensitivity.interest_rate_hike_impact}</span>
                      </div>
                      <div className="mt-2">
                        <span className="text-gray-500">Vulnerable to:</span>
                        <ul className="mt-1 space-y-1">
                          {application.risk_analysis.macroeconomic_sensitivity.most_vulnerable_to.map((factor, idx) => (
                            <li key={idx} className="text-sm text-gray-600 flex items-start">
                              <span className="text-red-500 mr-2">•</span>
                              {factor}
                            </li>
                          ))}
                        </ul>
                      </div>
                    </div>
                  </div>
                )}

                {/* What-If Scenarios */}
                {application.what_if_scenarios?.combined_recommendations?.length > 0 && (
                  <div className="border border-gray-200 rounded-lg p-4">
                    <h3 className="text-md font-medium text-gray-900 mb-3">Improvement Scenarios</h3>
                    <div className="space-y-3">
                      {application.what_if_scenarios.combined_recommendations.slice(0, 3).map((scenario) => (
                        <div key={scenario.id} className="p-3 bg-gray-50 rounded">
                          <h4 className="text-sm font-medium text-gray-900">{scenario.title}</h4>
                          <p className="text-xs text-gray-600 mt-1">{scenario.reasoning}</p>
                          <div className="mt-2 flex justify-between items-center">
                            <span className="text-xs text-gray-500">
                              {scenario.current_value} → {scenario.suggested_value}
                            </span>
                            <span className="text-xs bg-green-100 text-green-700 px-2 py-0.5 rounded">
                              +{scenario.improvement_percentage}%
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>

              {/* Files Processed */}
              {application.files_processed?.length > 0 && (
                <div className="border border-gray-200 rounded-lg p-4">
                  <h3 className="text-md font-medium text-gray-900 mb-3">Processed Files</h3>
                  <div className="space-y-2">
                    {application.files_processed.map((file, idx) => (
                      <div key={idx} className="flex items-center text-sm">
                        <svg className="w-4 h-4 text-gray-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                        </svg>
                        <span className="text-gray-600">{file}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-white">
        <div className="border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
            <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
          </div>
        </div>
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex justify-center items-center py-12">
            <div className="text-gray-400">Loading applications...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-2xl font-bold text-gray-900">Admin Dashboard</h1>
          <p className="text-sm text-gray-500">Monitor and analyze loan applications</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Stats Cards */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-8">
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="text-sm text-gray-500">Total Applications</div>
            <div className="text-2xl font-bold text-gray-900">{stats.total}</div>
          </div>
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="text-sm text-gray-500">High Risk</div>
            <div className="text-2xl font-bold text-red-600">{stats.highRisk}</div>
          </div>
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="text-sm text-gray-500">Medium Risk</div>
            <div className="text-2xl font-bold text-yellow-600">{stats.mediumRisk}</div>
          </div>
          <div className="border border-gray-200 rounded-lg p-4">
            <div className="text-sm text-gray-500">Low Risk</div>
            <div className="text-2xl font-bold text-green-600">{stats.lowRisk}</div>
          </div>
        </div>

        {/* Applications Table */}
        <div className="border border-gray-200 rounded-lg overflow-hidden">
          <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
            <h2 className="text-md font-medium text-gray-900">Recent Applications</h2>
          </div>
          
          {applications.length === 0 ? (
            <div className="p-8 text-center">
              <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-lg flex items-center justify-center">
                <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
                </svg>
              </div>
              <p className="text-gray-500">No applications found</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Date
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Application ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Risk Level
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Probability
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Files
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {applications.map((app) => (
                    <tr key={app.application_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {formatDate(app.created_at)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-mono text-gray-900">
                        {app.application_id.substring(0, 8)}...
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap">
                        <span className={`px-2 py-1 text-xs rounded-full ${
                          app.prediction_result?.risk_level?.toLowerCase().includes('high') ? 'bg-red-50 text-red-700' :
                          app.prediction_result?.risk_level?.toLowerCase().includes('medium') ? 'bg-yellow-50 text-yellow-700' :
                          'bg-green-50 text-green-700'
                        }`}>
                          {app.prediction_result?.risk_level || 'Unknown'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {(app.prediction_result?.probability * 100).toFixed(1)}%
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        {app.files_processed?.length || 0}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <button
                          onClick={() => {
                            setSelectedApp(app);
                            setShowDetailModal(true);
                          }}
                          className="text-gray-600 hover:text-gray-900"
                        >
                          View Details
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Error Message */}
        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-sm text-red-700">{error}</p>
          </div>
        )}
      </div>

      {/* Detail Modal */}
      {showDetailModal && selectedApp && (
        <DetailModal 
          application={selectedApp} 
          onClose={() => {
            setShowDetailModal(false);
            setSelectedApp(null);
          }} 
        />
      )}
    </div>
  );
}