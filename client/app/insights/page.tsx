'use client'
import { useState } from 'react';
import {
  BarChart, Bar, LineChart, Line, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, TooltipProps, Legend, ResponsiveContainer,
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, ComposedChart
} from 'recharts';
import { PieLabelRenderProps } from 'recharts';
import { NameType, ValueType, Payload } from 'recharts/types/component/DefaultTooltipContent';

// Credit Score Bands Data
const creditScoreBands = [
  { band: 'Poor (300-550)', count: 7999, approvalRate: 0.804, defaultRate: 0.333, color: '#ef4444' },
  { band: 'Fair (550-650)', count: 13651, approvalRate: 0.938, defaultRate: 0.178, color: '#f59e0b' },
  { band: 'Good (650-750)', count: 15533, approvalRate: 0.972, defaultRate: 0.102, color: '#10b981' },
  { band: 'Very Good (750-850)', count: 9301, approvalRate: 0.987, defaultRate: 0.044, color: '#06b6d4' },
  { band: 'Excellent (850+)', count: 2010, approvalRate: 0.993, defaultRate: 0.027, color: '#8b5cf6' }
];

// Feature Correlations
const featureCorrelations = [
  { feature: 'debt_to_income_ratio', correlation: -0.454, fullName: 'Debt to Income Ratio' },
  { feature: 'past_defaults_count', correlation: -0.281, fullName: 'Past Defaults Count' },
  { feature: 'monthly_emi_amount', correlation: -0.253, fullName: 'Monthly EMI Amount' },
  { feature: 'annuity_amount', correlation: -0.253, fullName: 'Annuity Amount' },
  { feature: 'credit_score', correlation: 0.244, fullName: 'Credit Score' },
  { feature: 'payment_punctuality_score', correlation: 0.230, fullName: 'Payment Punctuality' },
  { feature: 'external_risk_score_1', correlation: -0.222, fullName: 'External Risk Score' },
  { feature: 'past_default_flag', correlation: -0.219, fullName: 'Past Default Flag' },
  { feature: 'utility_bill_payment_score', correlation: 0.215, fullName: 'Utility Bill Payment' },
  { feature: 'interest_rate', correlation: -0.205, fullName: 'Interest Rate' }
];

// Feature Importance
const featureImportance = [
  { feature: 'employment_type', importance: 45.80, category: 'Employment' },
  { feature: 'debt_to_income_ratio', importance: 44.96, category: 'Debt' },
  { feature: 'past_default_flag', importance: 41.02, category: 'History' },
  { feature: 'past_defaults_count', importance: 39.68, category: 'History' },
  { feature: 'payment_punctuality_score', importance: 21.34, category: 'Behavior' },
  { feature: 'years_in_current_job', importance: 19.34, category: 'Employment' },
  { feature: 'credit_score', importance: 15.93, category: 'Credit' },
  { feature: 'income_volatility_index', importance: 15.36, category: 'Income' },
  { feature: 'salary_credit_regularity', importance: 13.84, category: 'Income' },
  { feature: 'monthly_income', importance: 13.32, category: 'Income' }
];

// LIME Explanations
const limeExplanations = [
  {
    id: 8180,
    prediction: 'APPROVED',
    confidence: 99.98,
    factors: [
      { feature: 'debt_to_income_ratio', impact: -0.0222, type: 'negative' },
      { feature: 'credit_score', impact: 0.0199, type: 'positive' },
      { feature: 'salary_credit_regularity', impact: 0.0147, type: 'positive' },
      { feature: 'monthly_income', impact: 0.0079, type: 'positive' },
      { feature: 'total_monthly_expenses', impact: 0.0075, type: 'positive' }
    ]
  },
  {
    id: 2857,
    prediction: 'APPROVED',
    confidence: 100.00,
    factors: [
      { feature: 'debt_to_income_ratio', impact: -0.0228, type: 'negative' },
      { feature: 'credit_score', impact: 0.0191, type: 'positive' },
      { feature: 'salary_credit_regularity', impact: 0.0136, type: 'positive' },
      { feature: 'monthly_income', impact: 0.0083, type: 'positive' },
      { feature: 'past_defaults_count', impact: -0.0042, type: 'negative' }
    ]
  },
  {
    id: 8873,
    prediction: 'APPROVED',
    confidence: 99.79,
    factors: [
      { feature: 'debt_to_income_ratio', impact: -0.0258, type: 'negative' },
      { feature: 'credit_score', impact: 0.0219, type: 'positive' },
      { feature: 'past_defaults_count', impact: -0.0072, type: 'negative' },
      { feature: 'loan_amount_requested', impact: -0.0054, type: 'negative' },
      { feature: 'occupation_type', impact: -0.0030, type: 'negative' }
    ]
  }
];

// Categorical Distributions
const categoricalData = {
  gender: [
    { name: 'Male', value: 27496, percentage: 55.0, color: '#3b82f6' },
    { name: 'Female', value: 22053, percentage: 44.1, color: '#ec4899' },
    { name: 'Other', value: 451, percentage: 0.9, color: '#8b5cf6' }
  ],
  maritalStatus: [
    { name: 'Married', value: 27431, percentage: 54.9, color: '#3b82f6' },
    { name: 'Single', value: 14988, percentage: 30.0, color: '#10b981' },
    { name: 'Divorced', value: 5523, percentage: 11.0, color: '#f59e0b' },
    { name: 'Widowed', value: 2058, percentage: 4.1, color: '#ef4444' }
  ],
  education: [
    { name: 'Graduate', value: 22641, percentage: 45.3, color: '#3b82f6' },
    { name: 'Under Graduate', value: 14831, percentage: 29.7, color: '#10b981' },
    { name: 'Post Graduate', value: 9908, percentage: 19.8, color: '#f59e0b' },
    { name: 'PhD', value: 2620, percentage: 5.2, color: '#8b5cf6' }
  ],
  loanPurpose: [
    { name: 'Personal', value: 14933, percentage: 29.9, color: '#ef4444' },
    { name: 'Home', value: 12454, percentage: 24.9, color: '#3b82f6' },
    { name: 'Business', value: 7527, percentage: 15.1, color: '#10b981' },
    { name: 'Car', value: 7505, percentage: 15.0, color: '#f59e0b' },
    { name: 'Education', value: 5016, percentage: 10.0, color: '#8b5cf6' }
  ],
  employment: [
    { name: 'Salaried', value: 31648, percentage: 63.3, color: '#3b82f6' },
    { name: 'Self Employed', value: 10555, percentage: 21.1, color: '#f59e0b' },
    { name: 'Gig', value: 4952, percentage: 9.9, color: '#10b981' },
    { name: 'Unemployed', value: 1670, percentage: 3.3, color: '#ef4444' },
    { name: 'Retired', value: 1175, percentage: 2.3, color: '#8b5cf6' }
  ]
};

// Model Performance Data
const modelPerformance = {
  metrics: [
    { name: 'Accuracy', value: 96.98 },
    { name: 'Precision', value: 97.87 },
    { name: 'Recall', value: 98.94 },
    { name: 'F1-Score', value: 98.40 },
    { name: 'ROC-AUC', value: 98.38 }
  ],
  cvFolds: [
    { name: 'Fold 1', value: 98.43 },
    { name: 'Fold 2', value: 98.60 },
    { name: 'Fold 3', value: 98.17 },
    { name: 'Fold 4', value: 98.53 },
    { name: 'Fold 5', value: 98.64 }
  ]
};


interface CustomTooltipProps {
  active?: boolean;
  payload?: Array<{
    name?: string;
    value?: number | string;
    color?: string;
    payload?: any;
  }>;
  label?: string;
}

// Custom Tooltip Component with proper typing
// Update the CustomTooltip component with proper typing
const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
  if (active && payload && payload.length) {
    return (
      <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm">
        <p className="text-sm font-medium text-gray-900">{label}</p>
        {payload.map((entry, index) => {
          // Safely handle different value types
          const value = entry.value;
          const formattedValue = typeof value === 'number' 
            ? value.toFixed(2) 
            : String(value ?? '');
          
          // Safely check if name contains 'Rate' - ensure it's a string
          const name = entry.name;
          const showPercent = typeof name === 'string' && name.includes('Rate');
          
          return (
            <p 
              key={index} 
              className="text-sm text-gray-600" 
              style={{ color: entry.color }}
            >
              {name}: {formattedValue}{showPercent ? '%' : ''}
            </p>
          );
        })}
      </div>
    );
  }
  return null;
};


// Custom Pie Label Component with proper typing
const renderCustomizedLabel = (props: PieLabelRenderProps) => {
  const { name, percent } = props;
  const safeName = name || '';
  const safePercent = typeof percent === 'number' ? percent : 0;
  return `${safeName}: ${(safePercent * 100).toFixed(1)}%`;
};

export default function InsightsPage() {
  const [selectedLime, setSelectedLime] = useState(limeExplanations[0]);
  const [activeTab, setActiveTab] = useState('overview');

  // Calculate category totals for feature importance
  const categoryTotals = featureImportance.reduce((acc, curr) => {
    acc[curr.category] = (acc[curr.category] || 0) + curr.importance;
    return acc;
  }, {} as Record<string, number>);

  const categoryData = Object.entries(categoryTotals).map(([name, value]) => ({
    name,
    value
  }));

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b border-gray-200 sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <h1 className="text-3xl font-bold text-gray-900">Model Insights Dashboard</h1>
          <p className="text-sm text-gray-500 mt-1">Comprehensive visualization of loan prediction model performance and data distributions</p>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            {['overview', 'distributions', 'features', 'explanations'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`py-4 px-1 text-sm font-medium border-b-2 capitalize transition-colors ${
                  activeTab === tab
                    ? 'border-blue-600 text-blue-600'
                    : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                }`}
              >
                {tab === 'overview' ? '📊 Model Overview' :
                 tab === 'distributions' ? '📈 Data Distributions' :
                 tab === 'features' ? '🔬 Feature Analysis' : '🤖 LIME Explanations'}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Overview Tab */}
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Model Performance Cards */}
            <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
              {modelPerformance.metrics.map((metric) => (
                <div key={metric.name} className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-sm transition-shadow">
                  <div className="text-sm text-gray-500 mb-2">{metric.name}</div>
                  <div className="text-3xl font-bold text-gray-900">{metric.value}%</div>
                  <div className="mt-4 h-1 bg-gray-100 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-blue-600 rounded-full"
                      style={{ width: `${metric.value}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>

            {/* Cross Validation and Credit Score Charts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Cross Validation */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">5-Fold Cross-Validation ROC-AUC</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={modelPerformance.cvFolds}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="name" stroke="#6b7280" />
                    <YAxis domain={[97, 99]} stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]}>
                      {modelPerformance.cvFolds.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={`rgba(59, 130, 246, ${0.6 + index * 0.1})`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
                <div className="mt-4 text-center text-sm text-gray-500">
                  Mean CV ROC-AUC: 98.47% (±0.34%)
                </div>
              </div>

              {/* Credit Score Distribution */}
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Credit Score Distribution</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={creditScoreBands}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      paddingAngle={2}
                      dataKey="count"
                      label={renderCustomizedLabel}
                    >
                      {creditScoreBands.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Credit Score Performance */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Credit Score Performance Analysis</h2>
              <ResponsiveContainer width="100%" height={350}>
                <ComposedChart data={creditScoreBands}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="band" stroke="#6b7280" angle={-45} textAnchor="end" height={80} />
                  <YAxis yAxisId="left" stroke="#6b7280" />
                  <YAxis yAxisId="right" orientation="right" stroke="#6b7280" />
                  <Tooltip content={<CustomTooltip />} />
                  <Legend />
                  <Bar yAxisId="left" dataKey="count" fill="#3b82f6" name="Count" barSize={30} />
                  <Line yAxisId="right" type="monotone" dataKey="approvalRate" stroke="#10b981" name="Approval Rate" strokeWidth={3} />
                  <Line yAxisId="right" type="monotone" dataKey="defaultRate" stroke="#ef4444" name="Default Rate" strokeWidth={3} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Distributions Tab */}
        {activeTab === 'distributions' && (
          <div className="space-y-8">
            {/* Gender and Marital Status */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Gender Distribution</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={categoricalData.gender}
                      cx="50%"
                      cy="50%"
                      outerRadius={100}
                      dataKey="value"
                      label={renderCustomizedLabel}
                    >
                      {categoricalData.gender.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Marital Status Distribution</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={categoricalData.maritalStatus}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="name" stroke="#6b7280" />
                    <YAxis stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {categoricalData.maritalStatus.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Education and Employment */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Education Level Distribution</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={categoricalData.education}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={100}
                      dataKey="value"
                      label={renderCustomizedLabel}
                    >
                      {categoricalData.education.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Employment Type Distribution</h2>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={categoricalData.employment} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" stroke="#6b7280" />
                    <YAxis type="category" dataKey="name" width={100} stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {categoricalData.employment.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Loan Purpose */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Loan Purpose Distribution</h2>
              <ResponsiveContainer width="100%" height={350}>
                <BarChart data={categoricalData.loanPurpose}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="name" stroke="#6b7280" />
                  <YAxis stroke="#6b7280" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                    {categoricalData.loanPurpose.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Features Tab */}
        {activeTab === 'features' && (
          <div className="space-y-8">
            {/* Feature Correlations */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Feature Correlations with Loan Approval</h2>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={featureCorrelations} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis type="number" domain={[-0.5, 0.5]} stroke="#6b7280" />
                  <YAxis type="category" dataKey="fullName" width={150} stroke="#6b7280" />
                  <Tooltip content={<CustomTooltip />} />
                  <Bar dataKey="correlation" radius={[0, 4, 4, 0]}>
                    {featureCorrelations.map((entry, index) => (
                      <Cell 
                        key={`cell-${index}`} 
                        fill={entry.correlation > 0 ? '#10b981' : '#ef4444'}
                        opacity={Math.abs(entry.correlation) * 2}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Feature Importance */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance (XGBoost Gain)</h2>
                <ResponsiveContainer width="100%" height={350}>
                  <BarChart data={featureImportance} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis type="number" stroke="#6b7280" />
                    <YAxis type="category" dataKey="feature" width={120} stroke="#6b7280" />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 p-6">
                <h2 className="text-lg font-semibold text-gray-900 mb-4">Feature Importance by Category</h2>
                <ResponsiveContainer width="100%" height={350}>
                  <PieChart>
                    <Pie
                      data={categoryData}
                      cx="50%"
                      cy="50%"
                      outerRadius={120}
                      dataKey="value"
                      label={renderCustomizedLabel}
                    >
                      {['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'].map((color, index) => (
                        <Cell key={`cell-${index}`} fill={color} />
                      ))}
                    </Pie>
                    <Tooltip content={<CustomTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Radar Chart */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">Top 8 Features Impact Radar</h2>
              <ResponsiveContainer width="100%" height={400}>
                <RadarChart outerRadius={150} data={featureImportance.slice(0, 8)}>
                  <PolarGrid stroke="#e5e7eb" />
                  <PolarAngleAxis dataKey="feature" stroke="#6b7280" />
                  <PolarRadiusAxis angle={30} domain={[0, 50]} stroke="#6b7280" />
                  <Radar name="Importance" dataKey="importance" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
                  <Tooltip content={<CustomTooltip />} />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {/* Explanations Tab */}
        {activeTab === 'explanations' && (
          <div className="space-y-8">
            {/* LIME Selector */}
            <div className="bg-white rounded-lg border border-gray-200 p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">LIME Local Explanations</h2>
              <div className="flex flex-wrap gap-2 mb-6">
                {limeExplanations.map((lime) => (
                  <button
                    key={lime.id}
                    onClick={() => setSelectedLime(lime)}
                    className={`px-4 py-2 text-sm font-medium rounded-lg transition-colors ${
                      selectedLime.id === lime.id
                        ? 'bg-blue-600 text-white'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    Sample #{lime.id}
                  </button>
                ))}
              </div>

              {/* Selected LIME Explanation */}
              <div className="bg-gray-50 rounded-lg p-6">
                <div className="flex justify-between items-center mb-6">
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">Sample #{selectedLime.id}</h3>
                    <p className="text-sm text-gray-500">Prediction Analysis</p>
                  </div>
                  <div className="bg-green-100 text-green-700 px-4 py-2 rounded-lg">
                    <span className="font-medium">{selectedLime.prediction}</span>
                    <span className="ml-2 text-sm">({selectedLime.confidence}% confidence)</span>
                  </div>
                </div>

                {/* Factor Impact Visualization */}
                <div className="space-y-4">
                  <h4 className="text-sm font-medium text-gray-700">Top Contributing Factors</h4>
                  {selectedLime.factors.map((factor, idx) => (
                    <div key={idx}>
                      <div className="flex justify-between text-sm mb-1">
                        <span className="text-gray-600">{factor.feature.replace(/_/g, ' ')}</span>
                        <span className={factor.type === 'positive' ? 'text-green-600' : 'text-red-600'}>
                          {factor.type === 'positive' ? '+' : ''}{factor.impact.toFixed(4)}
                        </span>
                      </div>
                      <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div 
                          className={`h-full ${factor.type === 'positive' ? 'bg-green-600' : 'bg-red-600'} rounded-full`}
                          style={{ width: `${Math.abs(factor.impact) * 1000}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>

                {/* Impact Summary */}
                <div className="mt-8 grid grid-cols-2 gap-4">
                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="text-sm font-medium text-green-700 mb-2">Positive Factors</div>
                    {selectedLime.factors.filter(f => f.type === 'positive').map((f, i) => (
                      <div key={i} className="text-xs text-green-600 flex justify-between">
                        <span>{f.feature.split('_')[0]}</span>
                        <span>+{f.impact.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                  <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="text-sm font-medium text-red-700 mb-2">Negative Factors</div>
                    {selectedLime.factors.filter(f => f.type === 'negative').map((f, i) => (
                      <div key={i} className="text-xs text-red-600 flex justify-between">
                        <span>{f.feature.split('_')[0]}</span>
                        <span>{f.impact.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}