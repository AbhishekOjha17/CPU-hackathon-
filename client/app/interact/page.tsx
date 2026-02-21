'use client'
import { useState, useEffect } from 'react';

// 15 Most Important Features (selected based on typical loan prediction models)
const IMPORTANT_FEATURES = [
  { 
    name: 'credit_score', 
    label: 'Credit Score', 
    min: 300, 
    max: 900, 
    step: 1,
    default: 650,
    description: 'Higher score increases approval chances'
  },
  { 
    name: 'loan_amount_requested', 
    label: 'Loan Amount (₹)', 
    min: 10000, 
    max: 10000000, 
    step: 10000,
    default: 500000,
    description: 'Lower amounts are easier to approve'
  },
  { 
    name: 'monthly_income', 
    label: 'Monthly Income (₹)', 
    min: 0, 
    max: 500000, 
    step: 5000,
    default: 50000,
    description: 'Higher income improves debt-to-income ratio'
  },
  { 
    name: 'existing_emi_obligations', 
    label: 'Existing EMI (₹)', 
    min: 0, 
    max: 200000, 
    step: 1000,
    default: 10000,
    description: 'Lower existing EMIs are better'
  },
  { 
    name: 'debt_to_income_ratio', 
    label: 'Debt-to-Income Ratio', 
    min: 0, 
    max: 1, 
    step: 0.01,
    default: 0.3,
    description: 'Lower ratio (under 0.4) is preferred'
  },
  { 
    name: 'past_defaults_count', 
    label: 'Past Defaults Count', 
    min: 0, 
    max: 5, 
    step: 1,
    default: 0,
    description: 'Zero defaults is ideal'
  },
  { 
    name: 'years_in_current_job', 
    label: 'Years in Current Job', 
    min: 0, 
    max: 30, 
    step: 0.5,
    default: 2,
    description: 'Longer job tenure indicates stability'
  },
  { 
    name: 'credit_history_length_years', 
    label: 'Credit History (Years)', 
    min: 0, 
    max: 30, 
    step: 0.5,
    default: 5,
    description: 'Longer history is better'
  },
  { 
    name: 'interest_rate', 
    label: 'Interest Rate (%)', 
    min: 5, 
    max: 25, 
    step: 0.1,
    default: 10.5,
    description: 'Lower rates reduce burden'
  },
  { 
    name: 'loan_term_months', 
    label: 'Loan Term (Months)', 
    min: 6, 
    max: 360, 
    step: 6,
    default: 60,
    description: 'Shorter terms have lower total interest'
  },
  { 
    name: 'applicant_age', 
    label: 'Applicant Age', 
    min: 18, 
    max: 80, 
    step: 1,
    default: 35,
    description: 'Middle age often preferred'
  },
  { 
    name: 'savings_balance', 
    label: 'Savings Balance (₹)', 
    min: 0, 
    max: 5000000, 
    step: 10000,
    default: 100000,
    description: 'Higher savings show financial strength'
  },
  { 
    name: 'credit_utilization_ratio', 
    label: 'Credit Utilization', 
    min: 0, 
    max: 1, 
    step: 0.01,
    default: 0.3,
    description: 'Under 0.3 is excellent'
  },
  { 
    name: 'dependents_count', 
    label: 'Number of Dependents', 
    min: 0, 
    max: 10, 
    step: 1,
    default: 1,
    description: 'Fewer dependents means more disposable income'
  },
  { 
    name: 'loan_to_value_ratio', 
    label: 'Loan-to-Value Ratio', 
    min: 0, 
    max: 1, 
    step: 0.01,
    default: 0.8,
    description: 'Lower ratio (under 0.8) is safer for lender'
  }
];

// Categorical features with their options
const CATEGORICAL_FEATURES = [
  {
    name: 'employment_type',
    label: 'Employment Type',
    options: ['Salaried', 'Self-Employed', 'Business Owner', 'Unemployed', 'Retired'],
    default: 'Salaried'
  },
  {
    name: 'education_level',
    label: 'Education Level',
    options: ['High School', 'Bachelor', 'Master', 'PhD', 'Other'],
    default: 'Bachelor'
  },
  {
    name: 'loan_purpose',
    label: 'Loan Purpose',
    options: ['Home', 'Car', 'Education', 'Personal', 'Business', 'Debt Consolidation'],
    default: 'Personal'
  },
  {
    name: 'marital_status',
    label: 'Marital Status',
    options: ['Single', 'Married', 'Divorced', 'Widowed'],
    default: 'Married'
  },
  {
    name: 'property_area',
    label: 'Property Area',
    options: ['Urban', 'Semi-Urban', 'Rural'],
    default: 'Urban'
  }
];

interface PredictionResult {
  risk_level: string;
  probability: number;
  confidence: number;
  risk_factors?: string[];
}

interface FeatureImpact {
  name: string;
  impact: number;
  direction: 'positive' | 'negative';
}

export default function InteractPage() {
  const [numericFeatures, setNumericFeatures] = useState<Record<string, number>>({});
  const [categoricalFeatures, setCategoricalFeatures] = useState<Record<string, string>>({});
  const [baseRecord, setBaseRecord] = useState<Record<string, any> | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [featureImpacts, setFeatureImpacts] = useState<FeatureImpact[]>([]);

  // Fetch a sample record from MongoDB on mount
  useEffect(() => {
    fetchBaseRecord();
  }, []);

  const fetchBaseRecord = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/applications?limit=1');
      const data = await response.json();
      
      if (data.success && data.applications.length > 0) {
        const record = data.applications[0].extracted_features;
        setBaseRecord(record);
        
        // Initialize numeric features with defaults from the record
        const initialNumeric: Record<string, number> = {};
        IMPORTANT_FEATURES.forEach(f => {
          initialNumeric[f.name] = record[f.name] || f.default;
        });
        setNumericFeatures(initialNumeric);
        
        // Initialize categorical features
        const initialCategorical: Record<string, string> = {};
        CATEGORICAL_FEATURES.forEach(f => {
          initialCategorical[f.name] = record[f.name] || f.default;
        });
        setCategoricalFeatures(initialCategorical);
      }
    } catch (err) {
      console.error('Failed to fetch base record:', err);
      // Initialize with defaults if fetch fails
      const initialNumeric: Record<string, number> = {};
      IMPORTANT_FEATURES.forEach(f => {
        initialNumeric[f.name] = f.default;
      });
      setNumericFeatures(initialNumeric);
      
      const initialCategorical: Record<string, string> = {};
      CATEGORICAL_FEATURES.forEach(f => {
        initialCategorical[f.name] = f.default;
      });
      setCategoricalFeatures(initialCategorical);
    }
  };

  const handleNumericChange = (name: string, value: number) => {
    setNumericFeatures(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleCategoricalChange = (name: string, value: string) => {
    setCategoricalFeatures(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      // Combine all features
      const allFeatures = {
        ...numericFeatures,
        ...categoricalFeatures,
        // Include other features from base record with defaults
        gender: baseRecord?.gender || 'Male',
        occupation_type: baseRecord?.occupation_type || 'Professional',
        has_credit_card_flag: baseRecord?.has_credit_card_flag || 1,
        income_verification_status: baseRecord?.income_verification_status || 'Verified',
        home_ownership_status: baseRecord?.home_ownership_status || 'Owned',
        application_channel: baseRecord?.application_channel || 'Online',
        region_economic_risk_score: baseRecord?.region_economic_risk_score || 0.5,
        urban_rural_indicator: baseRecord?.urban_rural_indicator || 'Urban',
        fraud_flag: baseRecord?.fraud_flag || 0,
        default_status: baseRecord?.default_status || 0
      };

      const response = await fetch('http://localhost:8000/api/predict-direct', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(allFeatures),
      });

      const data = await response.json();

      if (data.success) {
        setPrediction(data.prediction);
        
        // Generate feature impacts (simplified)
        const impacts: FeatureImpact[] = IMPORTANT_FEATURES.map(f => {
          const value = numericFeatures[f.name];
          let impact = 0;
          let direction: 'positive' | 'negative' = 'positive';
          
          // Simple rules for impact calculation
          if (f.name === 'credit_score') {
            impact = Math.min(0.3, (value - 300) / 2000);
            direction = value > 650 ? 'positive' : 'negative';
          } else if (f.name === 'debt_to_income_ratio') {
            impact = Math.min(0.25, value);
            direction = value < 0.4 ? 'positive' : 'negative';
          } else if (f.name === 'past_defaults_count') {
            impact = value * 0.1;
            direction = value === 0 ? 'positive' : 'negative';
          }
          
          return { name: f.label, impact, direction };
        }).sort((a, b) => b.impact - a.impact);
        
        setFeatureImpacts(impacts.slice(0, 5));
      } else {
        setError('Prediction failed');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to get prediction');
    } finally {
      setLoading(false);
    }
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

  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <div className="border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-2xl font-bold text-gray-900">Loan Prediction Model</h1>
          <p className="text-sm text-gray-500">Adjust parameters to see how they affect loan approval</p>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Column - Parameter Controls */}
          <div className="space-y-6">
            {/* Numeric Features */}
            <div className="border border-gray-200 rounded-lg">
              <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                <h2 className="text-md font-medium text-gray-900">Key Parameters</h2>
              </div>
              <div className="p-6 space-y-4">
                {IMPORTANT_FEATURES.map((feature) => (
                  <div key={feature.name}>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-700">
                        {feature.label}
                      </label>
                      <span className="text-sm text-gray-900">
                        {numericFeatures[feature.name]?.toLocaleString()}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={feature.min}
                      max={feature.max}
                      step={feature.step}
                      value={numericFeatures[feature.name] || feature.default}
                      onChange={(e) => handleNumericChange(feature.name, Number(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                    <p className="text-xs text-gray-500 mt-1">{feature.description}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* Categorical Features */}
            <div className="border border-gray-200 rounded-lg">
              <div className="px-6 py-4 bg-gray-50 border-b border-gray-200">
                <h2 className="text-md font-medium text-gray-900">Additional Details</h2>
              </div>
              <div className="p-6 space-y-4">
                {CATEGORICAL_FEATURES.map((feature) => (
                  <div key={feature.name}>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      {feature.label}
                    </label>
                    <select
                      value={categoricalFeatures[feature.name] || feature.default}
                      onChange={(e) => handleCategoricalChange(feature.name, e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded text-sm focus:outline-none focus:border-gray-500"
                    >
                      {feature.options.map((option) => (
                        <option key={option} value={option}>{option}</option>
                      ))}
                    </select>
                  </div>
                ))}
              </div>
            </div>

            {/* Predict Button */}
            <button
              onClick={handlePredict}
              disabled={loading}
              className={`w-full py-3 px-4 rounded font-medium text-white ${
                loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-gray-900 hover:bg-gray-800'
              }`}
            >
              {loading ? 'Calculating...' : 'Run Prediction'}
            </button>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                <p className="text-sm text-red-700">{error}</p>
              </div>
            )}
          </div>

          {/* Right Column - Results */}
          <div className="space-y-6">
            {prediction ? (
              <>
                {/* Prediction Result */}
                <div className={`p-6 rounded-lg border ${getRiskColor(prediction.risk_level)}`}>
                  <h3 className="text-lg font-medium mb-2">Prediction Result</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span>Risk Level:</span>
                      <span className="font-bold">{prediction.risk_level}</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Probability:</span>
                      <span>{(prediction.probability * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span>Confidence:</span>
                      <span>{(prediction.confidence * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                </div>

                {/* Feature Impact Analysis */}
                <div className="border border-gray-200 rounded-lg p-6">
                  <h3 className="text-md font-medium text-gray-900 mb-4">Top Factors Influencing Decision</h3>
                  <div className="space-y-3">
                    {featureImpacts.map((impact, idx) => (
                      <div key={idx}>
                        <div className="flex justify-between text-sm mb-1">
                          <span className="text-gray-600">{impact.name}</span>
                          <span className={`text-sm ${
                            impact.direction === 'positive' ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {impact.direction === 'positive' ? '+' : '-'}{(impact.impact * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                          <div 
                            className={`h-full rounded-full ${
                              impact.direction === 'positive' ? 'bg-green-600' : 'bg-red-600'
                            }`}
                            style={{ width: `${impact.impact * 100}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Recommendations based on risk level */}
                {prediction.risk_level?.toLowerCase() === 'high risk' && (
                  <div className="border border-gray-200 rounded-lg p-6">
                    <h3 className="text-md font-medium text-gray-900 mb-3">Recommendations</h3>
                    <ul className="space-y-2">
                      <li className="flex items-start text-sm text-gray-600">
                        <span className="text-red-500 mr-2">•</span>
                        Consider reducing loan amount or increasing down payment
                      </li>
                      <li className="flex items-start text-sm text-gray-600">
                        <span className="text-red-500 mr-2">•</span>
                        Work on improving credit score before applying
                      </li>
                      <li className="flex items-start text-sm text-gray-600">
                        <span className="text-red-500 mr-2">•</span>
                        Reduce existing debt to improve debt-to-income ratio
                      </li>
                    </ul>
                  </div>
                )}
              </>
            ) : (
              // Empty State
              <div className="border border-gray-200 rounded-lg p-8 text-center">
                <div className="w-16 h-16 mx-auto mb-4 bg-gray-100 rounded-lg flex items-center justify-center">
                  <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                </div>
                <h3 className="text-md font-medium text-gray-900 mb-2">No Prediction Yet</h3>
                <p className="text-sm text-gray-500">Adjust parameters and run prediction</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}