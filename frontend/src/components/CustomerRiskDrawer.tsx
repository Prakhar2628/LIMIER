"use client";

import { useState, useEffect } from "react";
import { ScoreResponse, CustomerRiskResponse, NarrativeResponse } from "@/lib/types";
import { fetchCustomerRisk, fetchCustomerNarrative } from "@/lib/api";
import { Sheet, SheetContent, SheetHeader, SheetTitle } from "./ui/sheet";
import { RiskBadge } from "./RiskBadge";
import { ShapFeatureChart } from "./ShapFeatureChart";
import {
  FileText,
  Sparkles,
  AlertCircle,
  Tag,
  Clock,
  BrainCircuit,
  Loader2,
  ChevronDown,
  ChevronUp,
  Activity,
} from "lucide-react";

interface CustomerRiskDrawerProps {
  customer: ScoreResponse | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function CustomerRiskDrawer({ customer, open, onOpenChange }: CustomerRiskDrawerProps) {
  const [details, setDetails] = useState<CustomerRiskResponse | ScoreResponse | null>(customer);
  const [loadingDetails, setLoadingDetails] = useState(false);

  // Generative AI narrative state
  const [narrative, setNarrative] = useState<string | null>(null);
  const [narrativeLoading, setNarrativeLoading] = useState(false);
  const [narrativeError, setNarrativeError] = useState(false);
  const [narrativeExpanded, setNarrativeExpanded] = useState(true);

  // Reset state when customer changes or drawer opens
  useEffect(() => {
    if (!customer?.customer_id || !open) return;
    setDetails(customer);
    setNarrative(null);
    setNarrativeError(false);

    async function loadFullDetails() {
      try {
        setLoadingDetails(true);
        const fullData = await fetchCustomerRisk(customer!.customer_id);
        setDetails(fullData);
      } catch (err) {
        console.error("Failed to load customer risk details:", err);
      } finally {
        setLoadingDetails(false);
      }
    }

    async function loadNarrative() {
      try {
        setNarrativeLoading(true);
        const resp: NarrativeResponse = await fetchCustomerNarrative(customer!.customer_id);
        setNarrative(resp.narrative);
      } catch (err) {
        console.error("Failed to load narrative:", err);
        setNarrativeError(true);
      } finally {
        setNarrativeLoading(false);
      }
    }

    loadFullDetails();
    loadNarrative();
  }, [customer?.customer_id, open]);

  if (!customer) return <Sheet open={open} onOpenChange={onOpenChange}><SheetContent><div/></SheetContent></Sheet>;

  const activeCustomer = details || customer;
  const topFeatures = activeCustomer.top_features || [];
  const nlp = (activeCustomer as CustomerRiskResponse).nlp_analysis;
  const recentTxnCount = (activeCustomer as CustomerRiskResponse).recent_transaction_count;
  const accountAgeDays = (activeCustomer as CustomerRiskResponse).account_age_days;

  const riskColor =
    activeCustomer.risk_level === "high"
      ? "text-red-400"
      : activeCustomer.risk_level === "medium"
      ? "text-amber-400"
      : "text-emerald-400";

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="overflow-y-auto w-full sm:max-w-xl bg-background/95 backdrop-blur-xl border-l border-white/10">
        {/* Header */}
        <SheetHeader className="mb-6 border-b border-white/10 pb-6">
          <SheetTitle className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="font-mono text-xl font-bold">{activeCustomer.customer_id}</span>
              {loadingDetails && (
                <span className="text-xs text-muted-foreground animate-pulse">(updating...)</span>
              )}
            </div>
            <RiskBadge level={activeCustomer.risk_level} />
          </SheetTitle>

          {/* Stats grid */}
          <div className="grid grid-cols-4 gap-3 text-sm mt-4 p-3 bg-white/5 rounded-xl border border-white/5">
            <div>
              <span className="block text-xs uppercase tracking-wider mb-1 text-muted-foreground">Final Score</span>
              <span className={`font-mono text-lg font-bold ${riskColor}`}>
                {typeof activeCustomer.final_score === "number"
                  ? activeCustomer.final_score.toFixed(1)
                  : "N/A"}
              </span>
            </div>
            <div>
              <span className="block text-xs uppercase tracking-wider mb-1 text-muted-foreground">ML Contrib</span>
              <span className="font-mono text-foreground text-lg font-bold">
                {typeof activeCustomer.ml_contribution === "number"
                  ? activeCustomer.ml_contribution.toFixed(3)
                  : "N/A"}
              </span>
            </div>
            <div>
              <span className="block text-xs uppercase tracking-wider mb-1 text-muted-foreground">Txn Count</span>
              <span className="font-mono text-foreground text-lg font-bold">
                {recentTxnCount ?? "—"}
              </span>
            </div>
            <div>
              <span className="block text-xs uppercase tracking-wider mb-1 text-muted-foreground">Acct Age</span>
              <span className="font-mono text-foreground text-lg font-bold">
                {accountAgeDays != null ? `${accountAgeDays}d` : "—"}
              </span>
            </div>
          </div>
        </SheetHeader>

        <div className="space-y-8 pb-8">

          {/* ── Generative AI Risk Narrative ────────────────────────── */}
          <section className="rounded-xl border border-violet-500/30 bg-violet-500/5 overflow-hidden">
            <button
              className="w-full flex items-center justify-between px-4 py-3 hover:bg-white/5 transition-colors"
              onClick={() => setNarrativeExpanded((v) => !v)}
            >
              <div className="flex items-center gap-2">
                <BrainCircuit className="w-4 h-4 text-violet-400" />
                <span className="text-sm font-semibold text-violet-300 uppercase tracking-wider">
                  Generative AI Risk Narrative
                </span>
              </div>
              <div className="flex items-center gap-2">
                {narrativeLoading && <Loader2 className="w-3.5 h-3.5 text-violet-400 animate-spin" />}
                {narrativeExpanded ? (
                  <ChevronUp className="w-4 h-4 text-muted-foreground" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-muted-foreground" />
                )}
              </div>
            </button>

            {narrativeExpanded && (
              <div className="px-4 pb-4">
                {narrativeLoading ? (
                  <div className="flex items-center gap-2 text-xs text-muted-foreground italic py-2">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-violet-400" />
                    Generating AI explanation via LLM...
                  </div>
                ) : narrativeError ? (
                  <div className="text-xs text-muted-foreground/60 italic py-2">
                    AI narrative unavailable. LLM provider may be unreachable.
                  </div>
                ) : narrative ? (
                  <p className="text-sm leading-relaxed text-foreground/90 border-l-2 border-violet-500/40 pl-3">
                    {narrative}
                  </p>
                ) : (
                  <div className="text-xs text-muted-foreground italic py-2">No narrative generated.</div>
                )}
              </div>
            )}
          </section>

          {/* ── NLP Text & Typology ─────────────────────────────────── */}
          <section className="glass-panel p-4 rounded-xl border border-primary/20 bg-primary/5">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-sm font-semibold flex items-center gap-2 text-primary uppercase tracking-wider">
                <Sparkles className="w-4 h-4 text-primary" />
                NLP Narrative &amp; Typology Analysis
              </h3>
              {nlp && (
                <span className="text-xs font-mono px-2 py-0.5 rounded bg-primary/20 text-primary border border-primary/30 font-bold">
                  NLP Score: {nlp.nlp_suspicious_score.toFixed(3)}
                </span>
              )}
            </div>

            {nlp ? (
              <div className="space-y-3 text-sm">
                <div className="flex items-center justify-between bg-black/30 p-2.5 rounded-lg border border-white/5">
                  <span className="text-xs text-muted-foreground uppercase">Predicted Typology</span>
                  <span className="font-medium text-foreground">{nlp.typology}</span>
                </div>

                <div className="flex items-center gap-4 bg-black/20 p-2.5 rounded-lg border border-white/5">
                  <div>
                    <span className="text-xs text-muted-foreground uppercase block mb-0.5">Sentiment</span>
                    <span className={`text-xs font-semibold uppercase ${
                      nlp.sentiment === "negative" ? "text-red-400"
                      : nlp.sentiment === "cautious" ? "text-amber-400"
                      : "text-emerald-400"
                    }`}>{nlp.sentiment}</span>
                  </div>
                  <div>
                    <span className="text-xs text-muted-foreground uppercase block mb-0.5">Urgency</span>
                    <span className={`text-xs font-semibold uppercase ${
                      nlp.urgency_level === "high" ? "text-red-400" : "text-muted-foreground"
                    }`}>{nlp.urgency_level}</span>
                  </div>
                </div>

                {nlp.detected_keywords && nlp.detected_keywords.length > 0 && (
                  <div>
                    <span className="text-xs text-muted-foreground block mb-1.5 uppercase">
                      Flagged Keywords &amp; Intents
                    </span>
                    <div className="flex flex-wrap gap-1.5">
                      {nlp.detected_keywords.map((kw, i) => (
                        <span
                          key={i}
                          className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-md bg-risk-high/10 text-risk-high border border-risk-high/20 font-mono"
                        >
                          <Tag className="w-3 h-3" />
                          {kw}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {nlp.sample_memos && nlp.sample_memos.length > 0 && (
                  <div>
                    <span className="text-xs text-muted-foreground block mb-1 uppercase">
                      Sample Transaction Notes
                    </span>
                    <div className="space-y-1.5">
                      {nlp.sample_memos.map((memo, i) => (
                        <div
                          key={i}
                          className="text-xs font-mono bg-black/40 p-2 rounded border border-white/5 text-muted-foreground italic flex items-start gap-2"
                        >
                          <FileText className="w-3.5 h-3.5 mt-0.5 text-primary shrink-0" />
                          <span>&ldquo;{memo}&rdquo;</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <div className="text-xs text-muted-foreground italic p-2">
                Analyzing transaction memos with NLP engine...
              </div>
            )}
          </section>

          {/* ── Triggered Rules ─────────────────────────────────────── */}
          <section>
            <h3 className="text-sm font-medium mb-3 text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-risk-high" />
              Triggered Deterministic Rules
            </h3>
            {activeCustomer.triggered_rules && activeCustomer.triggered_rules.length > 0 ? (
              <div className="flex flex-col gap-2">
                {activeCustomer.triggered_rules.map((rule, idx) => (
                  <div
                    key={idx}
                    className="glass-panel p-3 rounded-lg border border-white/5 bg-black/20"
                  >
                    <div className="font-medium text-sm text-foreground">{rule.rule}</div>
                    <div className="text-xs text-muted-foreground mt-1">{rule.reason}</div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-sm text-muted-foreground/50 italic p-3 glass-panel rounded-lg border border-white/5">
                No deterministic rules triggered for this entity.
              </div>
            )}
          </section>

          {/* ── Top Risk Features (SHAP / XAI) ─────────────────────── */}
          <section>
            <h3 className="text-sm font-medium mb-1 text-muted-foreground uppercase tracking-wider flex items-center gap-2">
              <Activity className="w-4 h-4 text-violet-400" />
              XAI Feature Attributions (SHAP)
            </h3>
            <p className="text-xs text-muted-foreground mb-3">
              Top features driving the XGBoost ensemble score.{" "}
              <span className="text-red-400">Red</span> = increases risk ·{" "}
              <span className="text-green-400">Green</span> = reduces risk.
            </p>
            {topFeatures && topFeatures.length > 0 ? (
              <div className="glass-panel p-4 rounded-lg border border-white/5 bg-black/20">
                <ShapFeatureChart features={topFeatures} />
              </div>
            ) : (
              <div className="text-sm text-muted-foreground/50 italic p-4 glass-panel rounded-lg border border-white/5 text-center">
                {loadingDetails ? (
                  <span className="flex items-center justify-center gap-2">
                    <Loader2 className="w-4 h-4 animate-spin" /> Computing feature breakdown...
                  </span>
                ) : (
                  "No SHAP features available for this customer."
                )}
              </div>
            )}
          </section>

        </div>
      </SheetContent>
    </Sheet>
  );
}
