"use client";

import { TopFeature } from "@/lib/types";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, ReferenceLine } from "recharts";

interface ShapFeatureChartProps {
  features: TopFeature[];
}

const FEATURE_LABEL_MAP: Record<string, string> = {
  iso_forest_score: "IF Anomaly Index",
  iso_forest_score_raw: "Raw IF Score",
  xgb_score: "XGBoost Fraud Proba",
  ensemble_score: "Meta-Ensemble Score",
  spike_ratio: "Volume Spike Ratio",
  spike_ratio_has_history: "Has Txn History",
  amount_rounded_to_nearest_100: "Round Amount Flag",
  distance_from_ctr_threshold: "CTR Proximity ($10k)",
  nlp_suspicious_score: "NLP Suspicious Index",
  rapid_cashout_ratio: "Rapid Cashout Ratio",
  structuring_proximity: "Structuring Proximity",
  log_amount: "Log Transaction Amount",
  cross_border_pct: "Cross-Border %",
  count_subthreshold_txns_7d: "7D Subthreshold Count",
};

const POSITIVE_COLOR = "#EF4444";   // red — increases fraud risk
const NEGATIVE_COLOR = "#22C55E";   // green — decreases fraud risk
const NEUTRAL_COLOR  = "#A78BFA";   // purple — neutral/inferred

// Friendly tooltip content
const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload || !payload.length) return null;
  const d = payload[0].payload;
  return (
    <div className="bg-[#12161A] border border-white/10 rounded-lg p-3 text-xs font-mono shadow-xl min-w-[200px]">
      <p className="text-white font-semibold mb-1">{d.name}</p>
      <div className="space-y-0.5 text-muted-foreground">
        <div className="flex justify-between gap-4">
          <span>Feature value:</span>
          <span className="text-white">{typeof d.value === "number" ? d.value.toFixed(4) : d.value}</span>
        </div>
        <div className="flex justify-between gap-4">
          <span>SHAP impact:</span>
          <span className={d.shap >= 0 ? "text-red-400" : "text-green-400"}>
            {d.shap !== 0 ? (d.shap > 0 ? "+" : "") + d.shap.toFixed(4) : "N/A"}
          </span>
        </div>
        <div className="flex justify-between gap-4">
          <span>Direction:</span>
          <span className={d.shap >= 0 ? "text-red-400" : "text-green-400"}>
            {d.shap > 0 ? "↑ Increases risk" : d.shap < 0 ? "↓ Reduces risk" : "Neutral"}
          </span>
        </div>
      </div>
    </div>
  );
};

export function ShapFeatureChart({ features }: ShapFeatureChartProps) {
  if (!features || features.length === 0) return null;

  const chartData = features.slice(0, 6).map((f, i) => {
    const rawName = f.feature || `Feature ${i + 1}`;
    const displayName =
      FEATURE_LABEL_MAP[rawName] ||
      rawName.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
    const shapVal = typeof f.shap_contribution === "number" ? f.shap_contribution : 0;
    const absShap = Math.abs(shapVal);
    // Use absolute SHAP for bar magnitude; fall back to scaled feature value
    const magnitude =
      absShap > 0.0001
        ? absShap
        : Math.min(1.0, Math.abs(f.value || 0.1) * 0.3 + 0.05);

    return {
      name: displayName,
      rawName,
      value: f.value ?? 0,
      shap: shapVal,
      magnitude: Math.max(0.03, magnitude),
      isPositive: shapVal >= 0,
      hasRealShap: absShap > 0.0001,
    };
  }).sort((a, b) => b.magnitude - a.magnitude);

  return (
    <div className="space-y-3">
      {/* Legend */}
      <div className="flex gap-4 text-xs text-muted-foreground px-1">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: POSITIVE_COLOR }} />
          Increases risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: NEGATIVE_COLOR }} />
          Reduces risk
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm inline-block" style={{ background: NEUTRAL_COLOR }} />
          Inferred
        </span>
      </div>

      {/* Bar chart */}
      <div className="h-56 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart
            data={chartData}
            layout="vertical"
            margin={{ top: 4, right: 36, left: 8, bottom: 4 }}
          >
            <XAxis type="number" hide domain={[0, "auto"]} />
            <YAxis
              dataKey="name"
              type="category"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "#a3a3a3", fontSize: 10 }}
              width={150}
            />
            <ReferenceLine x={0} stroke="rgba(255,255,255,0.1)" />
            <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
            <Bar dataKey="magnitude" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    !entry.hasRealShap
                      ? NEUTRAL_COLOR
                      : entry.isPositive
                      ? POSITIVE_COLOR
                      : NEGATIVE_COLOR
                  }
                  opacity={entry.hasRealShap ? 0.9 : 0.55}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Feature value table */}
      <div className="space-y-1 mt-1">
        {chartData.map((f, i) => (
          <div
            key={i}
            className="flex items-center justify-between text-xs px-2 py-1 rounded bg-white/[0.03] border border-white/5"
          >
            <span className="text-muted-foreground font-mono truncate max-w-[140px]">{f.name}</span>
            <div className="flex gap-3 font-mono">
              <span className="text-foreground/70">val: {typeof f.value === "number" ? f.value.toFixed(3) : "—"}</span>
              <span className={f.shap >= 0 ? "text-red-400" : "text-green-400"}>
                {f.hasRealShap
                  ? `SHAP: ${f.shap > 0 ? "+" : ""}${f.shap.toFixed(4)}`
                  : "SHAP: inferred"}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
