// Right-rail summary showing parsed labs and extracted report text.
function ReportPanel({ report, riskLevel }) {
  if (!report) {
    return (
      <div className="rounded-3xl border border-dashed border-white/10 bg-white/5 p-6 text-center text-white/60">
        Upload a PDF, image, or audio clip to parse labs and vitals.
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col gap-4 rounded-3xl border border-white/10 bg-white/5 p-4">
      <div>
        <p className="text-xs uppercase tracking-[0.4em] text-white/60">Report</p>
        <p className="mt-1 text-lg font-semibold text-white">{report.filename}</p>
        <p className="text-sm text-white/60">Risk level: {riskLevel}</p>
      </div>

      <div className="flex-1 rounded-2xl border border-white/10 bg-black/30 p-3 text-sm text-white/70">
        {report.extractedText}
      </div>

      {report.labs?.length > 0 && (
        <div>
          <p className="text-xs uppercase tracking-[0.4em] text-white/60">Parsed labs</p>
          <div className="mt-2 space-y-2">
            {report.labs.map((lab) => (
              <div
                key={lab.name}
                className="rounded-2xl border border-white/10 bg-black/40 px-3 py-2 text-sm text-white"
              >
                <div className="flex items-center justify-between">
                  <span className="font-semibold">{lab.name}</span>
                  <span className="text-white/70">{lab.value}</span>
                </div>
                <p className="text-xs text-white/50">Ref: {lab.range}</p>
                <p className="text-xs text-white/60">Flag: {lab.flag}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ReportPanel;
