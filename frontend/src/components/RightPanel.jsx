// Right panel showing uploaded reports and parsed labs

import { motion } from 'framer-motion';

function RightPanel({ report, riskLevel, collapsed, onToggleCollapse }) {
  if (collapsed) {
    return (
      <div className="glass-card flex w-16 flex-col items-center gap-4 p-4">
        <button
          onClick={onToggleCollapse}
          className="rounded-xl p-2 text-white/60 hover:bg-white/10 hover:text-white"
          aria-label="Expand panel"
        >
          →
        </button>
      </div>
    );
  }

  const getRiskBadgeClass = (level) => {
    switch (level) {
      case 'Red':
        return 'risk-badge-red';
      case 'Amber':
        return 'risk-badge-amber';
      default:
        return 'risk-badge-green';
    }
  };

  return (
    <aside className="glass-card flex w-96 flex-shrink-0 flex-col gap-4 p-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold text-white">Report Viewer</h3>
        <button
          onClick={onToggleCollapse}
          className="rounded-xl p-2 text-white/60 hover:bg-white/10 hover:text-white"
          aria-label="Collapse panel"
        >
          →
        </button>
      </div>

      {!report ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex flex-1 flex-col items-center justify-center rounded-2xl border-2 border-dashed border-white/10 p-8 text-center"
        >
          <div className="mb-4 text-5xl">📄</div>
          <p className="text-sm text-white/60">
            Upload a PDF, image, or audio file to parse labs and vitals
          </p>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          className="flex flex-1 flex-col gap-4"
        >
          {/* Report Header */}
          <div className="glass-card p-4">
            <div className="mb-2 flex items-start justify-between">
              <div className="flex-1">
                <p className="text-xs uppercase tracking-wider text-white/50">Document</p>
                <p className="mt-1 font-semibold text-white">{report.filename}</p>
                <p className="text-xs text-white/50">
                  {new Date(report.uploadedAt).toLocaleString()}
                </p>
              </div>
              <span className={getRiskBadgeClass(riskLevel)}>{riskLevel}</span>
            </div>
          </div>

          {/* Extracted Text Preview */}
          <div className="glass-card flex-1 overflow-y-auto p-4">
            <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">
              Extracted Text
            </p>
            <div className="whitespace-pre-wrap rounded-xl bg-black/30 p-3 text-xs text-white/80">
              {report.extractedText}
            </div>
          </div>

          {/* Parsed Labs */}
          {report.labs && report.labs.length > 0 && (
            <div className="glass-card p-4">
              <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-white/50">
                Parsed Labs
              </p>
              <div className="space-y-2">
                {report.labs.map((lab, idx) => (
                  <div
                    key={idx}
                    className="rounded-xl border border-white/10 bg-black/20 p-3 text-sm"
                  >
                    <div className="mb-1 flex items-center justify-between">
                      <span className="font-semibold text-white">{lab.name}</span>
                      <span className="text-white/70">{lab.value}</span>
                    </div>
                    <p className="text-xs text-white/50">Reference: {lab.range}</p>
                    <p
                      className={`mt-1 text-xs font-semibold ${
                        lab.flag === 'High' || lab.flag === 'Low'
                          ? 'text-danger'
                          : lab.flag === 'Borderline'
                          ? 'text-warning'
                          : 'text-success'
                      }`}
                    >
                      {lab.flag}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Risk Assessment */}
          {report.riskAssessment && (
            <div className="glass-card p-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-white/50">
                Risk Assessment
              </p>
              <p className="mb-3 text-sm text-white/80">{report.riskAssessment.reason}</p>
              {report.riskAssessment.recommendations && (
                <div>
                  <p className="mb-2 text-xs font-semibold text-white/70">Recommendations:</p>
                  <ul className="space-y-1 text-xs text-white/70">
                    {report.riskAssessment.recommendations.map((rec, idx) => (
                      <li key={idx} className="flex items-start gap-2">
                        <span className="text-primary">•</span>
                        <span>{rec}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Quick Actions */}
          <div className="flex gap-2">
            <button className="btn-secondary flex-1 text-sm">Export JSON</button>
            <button className="btn-secondary flex-1 text-sm">Share</button>
          </div>
        </motion.div>
      )}
    </aside>
  );
}

export default RightPanel;
