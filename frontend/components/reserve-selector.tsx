
"use client"

import React from "react"

import type { Reserve } from "@/lib/reserve-data"
import { MapPin, Trees, Waves, Mountain, Shrub } from "lucide-react"

const ecosystemIcons: Record<string, React.ComponentType<{ className?: string; style?: React.CSSProperties }>> = {
  "Coastal Sage Scrub / Maritime": Waves,
  "Chaparral / Riparian": Mountain,
  "Salt Marsh / Wetland": Waves,
  "Coastal Sage Scrub / Urban Canyon": Shrub,
}

interface ReserveSelectorProps {
  reserves: Reserve[]
  selectedReserve: string | null
  onSelectReserve: (id: string) => void
}

export function ReserveSelector({ reserves, selectedReserve, onSelectReserve }: ReserveSelectorProps) {
  return (
    <div className="flex flex-col gap-1.5">
      {reserves.map((reserve) => {
        const isSelected = selectedReserve === reserve.id
        const EcoIcon = ecosystemIcons[reserve.ecosystem] || Trees
        return (
          <button
            key={reserve.id}
            type="button"
            onClick={() => onSelectReserve(reserve.id)}
            className={`group flex items-center gap-3 rounded-lg px-3 py-2.5 text-left transition-all ${
              isSelected
                ? "bg-primary/10 ring-1 ring-primary/30"
                : "bg-card hover:bg-secondary"
            }`}
          >
            <div
              className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-lg transition-colors"
              style={{
                backgroundColor: isSelected ? `${reserve.color}22` : `${reserve.color}12`,
                border: `1px solid ${isSelected ? `${reserve.color}44` : "transparent"}`,
              }}
            >
              <EcoIcon
                className="h-4 w-4"
                style={{ color: reserve.color }}
              />
            </div>
            <div className="min-w-0 flex-1">
              <p className={`text-sm font-semibold leading-tight ${isSelected ? "text-foreground" : "text-foreground/80"}`}>
                {reserve.shortName}
              </p>
              <p className="mt-0.5 text-xs text-muted-foreground truncate">
                {reserve.ecosystem}
              </p>
            </div>
            <div className="flex flex-col items-end gap-0.5">
              <span className="text-xs font-semibold tabular-nums text-foreground/70">
                {reserve.totalSpecies}
              </span>
              <span className="text-[10px] text-muted-foreground">species</span>
            </div>
          </button>
        )
      })}
    </div>
  )
}

interface StatsBarProps {
  reserve: Reserve
}

export function StatsBar({ reserve }: StatsBarProps) {
  const observedTaxa = new Set(reserve.observedSpecies.map((s) => s.iconicTaxon))
  const predictedTaxa = new Set(reserve.predictedSpecies.map((s) => s.iconicTaxon))
  const avgProbability =
    reserve.predictedSpecies.length > 0
      ? reserve.predictedSpecies.reduce((acc, s) => acc + s.probability, 0) /
        reserve.predictedSpecies.length
      : 0

  const stats = [
    { label: "Total Observations", value: reserve.totalObservations.toLocaleString() },
    { label: "Observed Species", value: reserve.observedSpecies.length.toString() },
    { label: "Predicted Species", value: reserve.predictedSpecies.length.toString() },
    { label: "Taxa Groups", value: observedTaxa.size.toString() },
    { label: "Avg. Prediction", value: `${Math.round(avgProbability * 100)}%` },
    { label: "Area", value: `${reserve.areaHa} ha` },
  ]

  return (
    <div className="flex items-center gap-6 overflow-x-auto">
      {stats.map((stat) => (
        <div key={stat.label} className="flex flex-col items-center gap-0.5">
          <span className="text-lg font-bold tabular-nums text-foreground">{stat.value}</span>
          <span className="whitespace-nowrap text-[10px] font-medium uppercase tracking-wider text-muted-foreground">
            {stat.label}
          </span>
        </div>
      ))}
    </div>
  )
}
